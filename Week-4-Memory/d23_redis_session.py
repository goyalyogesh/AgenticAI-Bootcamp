"""
DEMO 23: Redis Session Persistence
------------------------------------
What you'll see:
  1. Store conversation history in Redis
  2. Retrieve it on a new server start (simulate restart)
  3. Expiry: sessions auto-delete after inactivity
  4. Cross-server sharing: multiple agents access same session

Prerequisites: Redis running (e.g. via Docker)
  docker run -d -p 6379:6379 redis:alpine

TTL in Production:
1. Customer Support - 24 Hours
2. Medical Records - Never (auto expire)
3. Short lived txns - 10 to 15 minutes
"""

import json
from datetime import datetime

import redis
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== REDIS SETUP =====
try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()
    print("✅ Redis connected")
    REDIS_AVAILABLE = True
except Exception:
    print("⚠️  Redis not available — using in-memory dict as fallback")
    REDIS_AVAILABLE = False
    r = {}  # fallback


# ===== REDIS-BACKED SESSION MANAGER =====
class RedisSessionManager:
    """
    Conversation sessions backed by Redis.
    Sessions expire after TTL seconds of inactivity.
    """

    SESSION_TTL = 3600  # 1 hour (in seconds)
    PREFIX = "session:"

    def __init__(self, redis_client):
        self.redis = redis_client
        self.sessions_in_memory = {}  # fallback

    def _key(self, session_id: str) -> str:
        return f"{self.PREFIX}{session_id}"

    def save_session(self, session_id: str, messages: list) -> None:
        """Serialize and save conversation to Redis."""
        serialized = [
            {"type": m.__class__.__name__, "content": m.content} for m in messages
        ]
        if REDIS_AVAILABLE:
            self.redis.setex(
                self._key(session_id),
                self.SESSION_TTL,
                json.dumps(serialized),
            )
        else:
            self.sessions_in_memory[session_id] = serialized
        print(f"  [Redis] Saved session {session_id} ({len(messages)} messages)")

    def load_session(self, session_id: str) -> list:
        """Load and deserialize conversation from Redis."""
        if REDIS_AVAILABLE:
            data = self.redis.get(self._key(session_id))
        else:
            data = self.sessions_in_memory.get(session_id)
            if data:
                data = json.dumps(data)

        if not data:
            print(f"  [Redis] No session found for {session_id} — starting fresh")
            return []

        serialized = json.loads(data)
        type_map = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage,
        }
        messages = [type_map[m["type"]](content=m["content"]) for m in serialized]
        print(f"  [Redis] Loaded session {session_id} ({len(messages)} messages)")
        return messages

    def delete_session(self, session_id: str) -> None:
        if REDIS_AVAILABLE:
            self.redis.delete(self._key(session_id))
        else:
            self.sessions_in_memory.pop(session_id, None)

    def extend_session(self, session_id: str) -> None:
        """Refresh session TTL on activity."""
        if REDIS_AVAILABLE:
            self.redis.expire(self._key(session_id), self.SESSION_TTL)


# ===== PERSISTENT CHAT AGENT =====
class PersistentChatAgent:
    """Agent whose conversations survive server restarts."""

    def __init__(self, system_prompt: str):
        self.session_manager = RedisSessionManager(r)
        self.system_prompt = system_prompt

    def chat(self, session_id: str, user_message: str) -> str:
        """Single turn of conversation — loads/saves from Redis."""
        # Load existing session (or empty list if new)
        history = self.session_manager.load_session(session_id)

        # Build context
        context = [SystemMessage(content=self.system_prompt)]
        context.extend(history)
        context.append(HumanMessage(content=user_message))

        # Call LLM
        response = llm.invoke(context)

        # Save updated history
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=response.content))
        self.session_manager.save_session(session_id, history)
        self.session_manager.extend_session(session_id)

        return response.content


if __name__ == "__main__":
   

    SESSION_ID = "demo_2"
    system_prompt = "You are a helpful TechShop support agent."

    print("=" * 60)
    print("DEMO 23: Redis Session Persistence")
    print("=" * 60)
    print("Choose mode:")
    print("  1) Run canned server-restart demo")
    print("  2) Interactive chat with persistent session")
    choice = input("Enter 1 or 2 (default 1): ").strip() or "1"

    if choice == "1":
        # ===== DEMO: Simulate server restart =====
        agent = PersistentChatAgent(system_prompt)

        print("\nSERVER 1 INSTANCE:")
        print("=" * 50)
        print("user message: Hi, my order ORD-555 hasn't arrived.")
        r1 = agent.chat(SESSION_ID, "Hi, my order ORD-555 hasn't arrived.")
        print(f"Turn 1: {r1[:100]}...")

        print("user message: I ordered it 7 days ago. Should I be worried?")
        r2 = agent.chat(SESSION_ID, "I ordered it 7 days ago. Should I be worried?")
        print(f"Turn 2: {r2[:100]}...")

        # ===== SIMULATE SERVER RESTART =====
        print("\n🔄 SERVER RESTART SIMULATED...")
        print("Creating NEW agent instance (simulates new server process)")
        print()

        new_agent = PersistentChatAgent(system_prompt)

        print("SERVER 2 INSTANCE (after restart):")
        print("=" * 50)
        print("user message: Can you remind me of my order number again?")
        r3 = new_agent.chat(
            SESSION_ID, "Can you remind me of my order number again?"
        )
        print(f"Turn 3 (new server): {r3[:150]}...")

        print("\n✅ Memory survived server restart!")
        print("   The new server loaded the session from Redis and knows about ORD-555")

    else:
        # ===== INTERACTIVE MODE =====
        agent = PersistentChatAgent(system_prompt)
        print("\nEntering interactive mode.")
        print(f"Using session_id = {SESSION_ID!r}")
        print("Type messages and press Enter. Commands:")
        print("  /exit  → quit")
        print()

        while True:
            msg = input("You: ").strip()
            if not msg:
                continue
            if msg.lower() in {"/exit", "exit", "quit"}:
                print("Exiting interactive Redis demo.")
                break

            reply = agent.chat(SESSION_ID, msg)
            print(f"Agent: {reply}\n")

