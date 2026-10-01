import sys
from orchestrator import Orchestrator

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <user_id> <query>")
        print("Example: python3 main.py analyst_user 'What is machine learning?'")
        print("\nAvailable users: admin_user, analyst_user, viewer_user")
        sys.exit(1)

    user_id = sys.argv[1]
    query = " ".join(sys.argv[2:])

    orch = Orchestrator()
    result = orch.process(query, user_id)

    if result["success"]:
        print(f"\nResponse:\n{result['response']}")
    else:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    main()
