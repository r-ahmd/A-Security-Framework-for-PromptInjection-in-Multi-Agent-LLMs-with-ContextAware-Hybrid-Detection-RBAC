"""
cli.py - Command-line tool for prompt injection detection and access control
"""

import sys
import os
import argparse
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("threat_agent").setLevel(logging.ERROR)
logging.getLogger("ml_classifier").setLevel(logging.WARNING)
logging.getLogger("transformer_classifier").setLevel(logging.WARNING)

from agents.threat_agent import ThreatAgent
from ml_classifier import MLPromptClassifier
from transformer_classifier import TransformerClassifier
from agent_access_control import get_default_access_control, ActionType


def setup_detectors():
    pattern = ThreatAgent()
    ml = MLPromptClassifier()
    transformer = TransformerClassifier()

    try:
        ml.load_model("ml_classifier.pkl")
    except Exception:
        pass
    try:
        transformer.load_model("transformer_classifier")
    except Exception:
        pass

    detectors = {"pattern": pattern.process}
    if ml.is_trained:
        detectors["ml"] = ml.predict
    if transformer.is_trained:
        detectors["transformer"] = transformer.predict

    return detectors


def detect_text(text, detectors, show_all=False):
    results = {}
    for name, func in detectors.items():
        results[name] = func(text)

    any_unsafe = any(r.get("safe") is False for r in results.values())

    if show_all:
        for name, r in results.items():
            status = "\033[31mUNSAFE\033[0m" if not r.get("safe", True) else "\033[32mSAFE\033[0m"
            print(f"  [{status}] {name}: score={r.get('score', 0):.4f}")

    return results, not any_unsafe


def cmd_detect(args):
    text = ""
    if args.file:
        with open(args.file, "r") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text

    detectors = setup_detectors()
    results, safe = detect_text(text, detectors, show_all=args.all or not args.json)

    if args.json:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.bool_, np.integer)):
                    return bool(o) if isinstance(o, np.bool_) else int(o)
                return super().default(o)
        print(json.dumps({"text": text, "safe": safe, "results": results}, cls=NumpyEncoder, indent=2))
    else:
        status = "\033[32mSAFE\033[0m" if safe else "\033[31mUNSAFE\033[0m"
        print(f"\nOverall: {status}")
        if not args.all:
            print("  (use -a to see per-detector breakdown)")


def cmd_agents(args):
    ac = get_default_access_control()
    agents = ac.list_agents()

    if args.json:
        print(json.dumps(agents, indent=2))
        return

    print(f"\n{'Agent':<20} {'Role':<18} {'Permissions':<40}")
    print("-" * 78)
    for a in agents:
        perms = ", ".join(a["permissions"])
        print(f"{a['name']:<20} {a['role']:<18} {perms:<40}")
    print(f"\nTotal: {len(agents)} agent(s)")


def cmd_roles(args):
    ac = get_default_access_control()
    roles = ac.list_roles()

    if args.json:
        print(json.dumps(roles, indent=2))
        return

    print(f"\n{'Role':<18} {'Permissions':<60}")
    print("-" * 78)
    for r in roles:
        perms = ", ".join(r["permissions"])
        print(f"{r['name']:<18} {perms:<60}")
    print(f"\nTotal: {len(roles)} role(s)")


def cmd_authorize(args):
    ac = get_default_access_control()

    try:
        action_type = ActionType(args.action)
    except ValueError:
        print(f"\033[31mError:\033[0m Unknown action: {args.action}")
        print(f"  Valid actions: {', '.join(a.value for a in ActionType)}")
        sys.exit(1)

    detection_safe = True
    if args.detect:
        detectors = setup_detectors()
        det_results, detection_safe = detect_text(args.detect, detectors)
    else:
        det_results = {}

    decision = ac.authorize_request(args.agent, action_type,
                                    detection_result={"safe": detection_safe, "score": 0.0} if args.detect else None)

    if args.json:
        print(json.dumps({
            "allowed": decision.allowed,
            "reason": decision.reason,
            "agent": decision.agent,
            "action": decision.action.value,
            "role": decision.role,
            "detection_safe": detection_safe,
        }, indent=2))
    else:
        status = "\033[32mALLOW\033[0m" if decision.allowed else "\033[31mDENY\033[0m"
        print(f"\n  [{status}] {decision.agent}.{decision.action.value}")
        print(f"  Role:      {decision.role}")
        print(f"  Reason:    {decision.reason}")
        if args.detect:
            det_status = "\033[32mSAFE\033[0m" if detection_safe else "\033[31mUNSAFE\033[0m"
            print(f"  Detection: {det_status}")


def main():
    parser = argparse.ArgumentParser(description="Prompt Injection Detection & Access Control CLI")

    sub = parser.add_subparsers(title="commands", dest="command")

    p_detect = sub.add_parser("detect", help="Analyze text for prompt injection")
    p_detect.add_argument("text", help="Text to analyze")
    p_detect.add_argument("-f", "--file", help="File containing text to analyze")
    p_detect.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    p_detect.add_argument("-a", "--all", action="store_true", help="Show all detectors")
    p_detect.add_argument("--detectors", nargs="+", default=["pattern", "ml", "transformer"],
                          help="Detectors to use")

    p_agents = sub.add_parser("agents", help="List agents and their roles/permissions")
    p_agents.add_argument("-j", "--json", action="store_true")

    p_roles = sub.add_parser("roles", help="List roles and their permissions")
    p_roles.add_argument("-j", "--json", action="store_true")

    p_auth = sub.add_parser("authorize", help="Check if an agent can perform an action")
    p_auth.add_argument("agent", help="Agent name")
    p_auth.add_argument("action", help="Action to check")
    p_auth.add_argument("--detect", help="Optional prompt to run injection detection on")
    p_auth.add_argument("-j", "--json", action="store_true")

    args = parser.parse_args()

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "agents":
        cmd_agents(args)
    elif args.command == "roles":
        cmd_roles(args)
    elif args.command == "authorize":
        cmd_authorize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()