import json
from utils.logger import setup_logger

logger = setup_logger("rbac_agent")

class RBACAgent:
    def __init__(self, config_path: str = "config/rbac_policy.json"):
        with open(config_path, "r") as f:
            self.policy = json.load(f)

    def process(self, user_id: str, action: str) -> dict:
        role = self.policy["users"].get(user_id)
        if not role:
            logger.warning(f"Unknown user: {user_id}")
            return {"authorized": False, "role": "none", "reason": "Unknown user"}

        permissions = self.policy["roles"].get(role, {}).get("permissions", [])
        authorized = action in permissions

        if authorized:
            logger.info(f"Authorized: {user_id} ({role}) -> {action}")
        else:
            logger.warning(f"Denied: {user_id} ({role}) -> {action}")

        return {
            "authorized": authorized,
            "role": role,
            "reason": f"Action '{action}' {'allowed' if authorized else 'denied'} for role '{role}'"
        }
