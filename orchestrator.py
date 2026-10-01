from agents.query_agent import QueryAgent
from agents.threat_agent import ThreatAgent
from agents.rbac_agent import RBACAgent
from agents.response_agent import ResponseAgent
from utils.logger import setup_logger

logger = setup_logger("orchestrator")

class Orchestrator:
    def __init__(self):
        self.query_agent = QueryAgent()
        self.threat_agent = ThreatAgent()
        self.rbac_agent = RBACAgent()
        self.response_agent = ResponseAgent()

    def process(self, user_query: str, user_id: str) -> dict:
        logger.info(f"Processing request from {user_id}: {user_query[:50]}...")

        threat_result = self.threat_agent.process(user_query)
        if not threat_result["safe"]:
            logger.warning(f"Request blocked: {threat_result['reason']}")
            return {"success": False, "response": None, "error": f"Threat detected: {threat_result['reason']}"}

        action = "execute"
        rbac_result = self.rbac_agent.process(user_id, action)
        if not rbac_result["authorized"]:
            logger.warning(f"Unauthorized: {rbac_result['reason']}")
            return {"success": False, "response": None, "error": f"Unauthorized: {rbac_result['reason']}"}

        query_result = self.query_agent.process(user_query, user_id)

        response_result = self.response_agent.process(query_result["query"])

        logger.info(f"Request completed successfully for {user_id}")
        return {"success": True, "response": response_result["response"], "error": None}
