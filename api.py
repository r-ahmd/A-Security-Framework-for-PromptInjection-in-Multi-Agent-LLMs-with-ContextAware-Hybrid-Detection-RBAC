"""
api.py - FastAPI server for prompt injection detection and access control
Provides REST endpoints for detection, authorization, and agent management
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("threat_agent").setLevel(logging.ERROR)
logging.getLogger("ml_classifier").setLevel(logging.WARNING)
logging.getLogger("transformer_classifier").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.threat_agent import ThreatAgent
from ml_classifier import MLPromptClassifier
from transformer_classifier import TransformerClassifier
from agent_access_control import (
    get_default_access_control, ActionType, AccessControl
)

app = FastAPI(
    title="Prompt Injection Detection & Access Control API",
    description="Multi-detector API with RBAC for multi-agent LLM security",
    version="2.0.0",
)

pattern_agent = ThreatAgent()
ml_classifier = MLPromptClassifier()
transformer_clf = TransformerClassifier()
access_control = get_default_access_control()

try:
    ml_classifier.load_model("ml_classifier.pkl")
except Exception:
    pass

try:
    transformer_clf.load_model("transformer_classifier")
except Exception:
    pass


class DetectRequest(BaseModel):
    text: str
    detectors: list[str] = ["pattern", "ml", "transformer"]


class DetectResponse(BaseModel):
    text: str
    results: dict
    overall_safe: bool
    latency_ms: float


class AuthorizeRequest(BaseModel):
    agent: str
    action: str
    prompt: str = ""
    detectors: list[str] = ["pattern"]


class AuthorizeResponse(BaseModel):
    allowed: bool
    reason: str
    agent: str
    action: str
    role: str
    detection_safe: bool
    detection_results: dict | None = None


class AgentInfo(BaseModel):
    name: str
    role: str
    description: str
    permissions: list[str]


@app.get("/")
def root():
    return {
        "service": "Prompt Injection Detection & Access Control API",
        "detectors": ["pattern", "ml", "transformer"],
        "access_control": True,
        "agents": len(access_control.agents),
        "roles": len(access_control.roles),
        "ml_trained": ml_classifier.is_trained,
        "transformer_trained": transformer_clf.is_trained,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.perf_counter()
    results = {}

    available = {
        "pattern": lambda t: pattern_agent.process(t),
    }
    if ml_classifier.is_trained:
        available["ml"] = lambda t: ml_classifier.predict(t)
    if transformer_clf.is_trained:
        available["transformer"] = lambda t: transformer_clf.predict(t)

    for name in req.detectors:
        if name not in available:
            results[name] = {"error": f"Detector '{name}' not available"}
            continue
        try:
            result = available[name](req.text)
            result.pop("reason", None)
            for k, v in result.items():
                if hasattr(v, "item"):
                    result[k] = v.item()
            results[name] = result
        except Exception as e:
            results[name] = {"error": str(e), "safe": True, "score": 0.0}

    any_unsafe = any(
        r.get("safe", True) == False for r in results.values()
    )
    elapsed = (time.perf_counter() - start) * 1000

    return DetectResponse(
        text=req.text,
        results=results,
        overall_safe=not any_unsafe,
        latency_ms=round(elapsed, 2),
    )


@app.post("/detect/batch")
def detect_batch(reqs: list[DetectRequest]):
    return [detect(req) for req in reqs]


@app.post("/authorize", response_model=AuthorizeResponse)
def authorize(req: AuthorizeRequest):
    try:
        action_type = ActionType(req.action)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    detection_results = {}
    detection_safe = True

    if req.prompt.strip():
        for name in req.detectors:
            if name == "pattern":
                detection_results[name] = pattern_agent.process(req.prompt)
            elif name == "ml" and ml_classifier.is_trained:
                detection_results[name] = ml_classifier.predict(req.prompt)
            elif name == "transformer" and transformer_clf.is_trained:
                detection_results[name] = transformer_clf.predict(req.prompt)

        for res_dict in detection_results.values():
            for k, v in res_dict.items():
                if hasattr(v, "item"):
                    res_dict[k] = v.item()

        detection_safe = all(
            r.get("safe", True) for r in detection_results.values()
        )

    decision = access_control.authorize_request(
        req.agent, action_type,
        detection_result={"safe": detection_safe, "score": 0.0}
    )

    return AuthorizeResponse(
        allowed=decision.allowed and detection_safe,
        reason=decision.reason,
        agent=req.agent,
        action=req.action,
        role=decision.role,
        detection_safe=detection_safe,
        detection_results=detection_results if detection_results else None,
    )


@app.get("/agents", response_model=list[AgentInfo])
def list_agents():
    return access_control.list_agents()


@app.get("/roles")
def list_roles():
    return access_control.list_roles()


# ── Ollama (Llama 3) Integration ─────────────────────────────────
# Full multi-agent pipeline: Detection → RBAC → LLM Response

import requests as http_requests
from agents.query_agent import QueryAgent
from agents.response_agent import ResponseAgent

query_agent = QueryAgent()
response_agent = ResponseAgent(model="llama3", base_url="http://localhost:11434")


class ChatRequest(BaseModel):
    prompt: str
    agent: str = "WorkerAgent"
    detectors: list[str] = ["pattern", "ml", "transformer"]


class ChatResponse(BaseModel):
    prompt: str
    llm_response: str | None
    blocked: bool
    pipeline: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full multi-agent security pipeline:
      Gate 1: Prompt Injection Detection (Pattern + ML + Transformer)
      Gate 2: Role-Based Access Control (RBAC)
      Gate 3: Forward safe prompt to Ollama Llama 3 for response
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    start = time.perf_counter()
    pipeline = {
        "gate1_detection": {},
        "gate2_rbac": {},
        "gate3_llm": {},
        "total_latency_ms": 0.0,
    }

    # ── Gate 1: Prompt Injection Detection ──
    detection_results = {}
    available = {"pattern": lambda t: pattern_agent.process(t)}
    if ml_classifier.is_trained:
        available["ml"] = lambda t: ml_classifier.predict(t)
    if transformer_clf.is_trained:
        available["transformer"] = lambda t: transformer_clf.predict(t)

    for name in req.detectors:
        if name in available:
            try:
                result = available[name](req.prompt)
                result.pop("reason", None)
                for k, v in result.items():
                    if hasattr(v, "item"):
                        result[k] = v.item()
                detection_results[name] = result
            except Exception as e:
                detection_results[name] = {"error": str(e), "safe": True, "score": 0.0}

    detection_safe = all(r.get("safe", True) for r in detection_results.values())
    pipeline["gate1_detection"] = {
        "safe": detection_safe,
        "detectors": detection_results,
    }

    if not detection_safe:
        pipeline["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(
            prompt=req.prompt,
            llm_response=None,
            blocked=True,
            pipeline=pipeline,
        )

    # ── Gate 2: RBAC Authorization ──
    try:
        action_type = ActionType("execute")
        decision = access_control.authorize_request(
            req.agent, action_type,
            detection_result={"safe": True, "score": 0.0}
        )
        pipeline["gate2_rbac"] = {
            "agent": req.agent,
            "role": decision.role,
            "allowed": decision.allowed,
            "reason": decision.reason,
        }
    except Exception as e:
        pipeline["gate2_rbac"] = {
            "agent": req.agent,
            "allowed": False,
            "reason": str(e),
        }
        pipeline["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(
            prompt=req.prompt,
            llm_response=None,
            blocked=True,
            pipeline=pipeline,
        )

    if not pipeline["gate2_rbac"].get("allowed", False):
        pipeline["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(
            prompt=req.prompt,
            llm_response=None,
            blocked=True,
            pipeline=pipeline,
        )

    # ── Gate 3: Forward to Ollama Llama 3 ──
    try:
        llm_result = response_agent.process(req.prompt)
        pipeline["gate3_llm"] = {
            "model": llm_result.get("model", "llama3"),
            "response_length": len(llm_result.get("response", "")),
            "status": "success",
        }
        llm_response = llm_result["response"]
    except Exception as e:
        pipeline["gate3_llm"] = {"status": "error", "error": str(e)}
        llm_response = f"LLM Error: {str(e)}"

    pipeline["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

    return ChatResponse(
        prompt=req.prompt,
        llm_response=llm_response,
        blocked=False,
        pipeline=pipeline,
    )


@app.get("/ollama/status")
def ollama_status():
    """Check if Ollama is running and which models are available."""
    try:
        resp = http_requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "online", "models": models}
    except Exception:
        return {"status": "offline", "models": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
