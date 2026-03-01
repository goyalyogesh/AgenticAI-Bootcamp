"""
Memory Types Comparison
----------------------------------
What you'll see:
  1. Buffer memory — full history, context preservation
  2. Summary memory — compressed history
  3. Token-limited sliding window
  4. Entity extraction memory
  5. Practical comparison on same conversation

Key concepts: memory strategies, context window management, entity extraction

use /stats to show the stats of the memory types
use /exit to exit the program
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== BUFFER MEMORY (Manual Implementation) =====
class BufferMemory:
    """Keep ALL messages. Simple but grows without bound."""

    def __init__(self):
        self.messages: list = []
        self.system_prompt = "You are a helpful assistant."

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def get_context(self) -> list:
        return [SystemMessage(content=self.system_prompt)] + self.messages

    def stats(self) -> dict:
        total_chars = sum(len(m.content) for m in self.messages)
        return {"messages": len(self.messages), "total_chars": total_chars}


# ===== SLIDING WINDOW MEMORY =====
class SlidingWindowMemory:
    """Keep only the last N messages. Fixed cost regardless of conversation length."""

    def __init__(self, window_size: int = 6):
        self.all_messages: list = []
        self.window_size = window_size
        self.system_prompt = "You are a helpful assistant."

    def add_user(self, text: str) -> None:
        self.all_messages.append(HumanMessage(content=text))

    def add_assistant(self, text: str) -> None:
        self.all_messages.append(AIMessage(content=text))

    def get_context(self) -> list:
        # Always keep system prompt + last N messages
        recent = self.all_messages[-self.window_size :]
        return [SystemMessage(content=self.system_prompt)] + recent

    def stats(self) -> dict:
        return {
            "total_messages": len(self.all_messages),
            "active_window": min(self.window_size, len(self.all_messages)),
        }


# ===== SUMMARY MEMORY =====
class SummaryMemory:
    """Compress old messages into a summary every N turns."""

    def __init__(self, summarize_after: int = 4):
        self.recent_messages: list = []
        self.summary: str = ""
        self.summarize_after = summarize_after

    def add_user(self, text: str) -> None:
        self.recent_messages.append(HumanMessage(content=text))

    def add_assistant(self, text: str) -> None:
        self.recent_messages.append(AIMessage(content=text))
        if len(self.recent_messages) >= self.summarize_after * 2:
            self._compress()

    def _compress(self) -> None:
        """Use LLM to summarize the conversation so far."""
        compress_messages = [
            SystemMessage(
                content=(
                    "Summarize this conversation in 2-3 sentences. "
                    "Focus on: topics discussed, decisions made, key information shared."
                )
            ),
            HumanMessage(
                content="\n".join(
                    [f"{m.type}: {m.content}" for m in self.recent_messages]
                )
            ),
        ]
        summary_response = llm.invoke(compress_messages)
        self.summary = summary_response.content
        # Keep only the last 2 messages after compression
        self.recent_messages = self.recent_messages[-2:]
        print(f"  [SummaryMemory] Compressed to: {self.summary[:80]}...")

    def get_context(self) -> list:
        context = [SystemMessage(content="You are a helpful assistant.")]
        if self.summary:
            context.append(
                SystemMessage(
                    content=f"Conversation summary so far: {self.summary}"
                )
            )
        context.extend(self.recent_messages)
        return context


# ===== ENTITY MEMORY =====
class EntityMemory:
    """Extract and track named entities (orders, users, products) from conversation."""

    def __init__(self):
        self.entities: dict = {}  # entity_name → description
        self.messages: list = []

    def extract_entities(self, text: str) -> None:
        """Use LLM to extract entities from latest message."""
        extract_messages = [
            SystemMessage(
                content=(
                    "Extract named entities from this text.\n"
                    'Return JSON with format: {"entities": {"entity_name": "what was said about it"}}\n'
                    "Only include specific named things (order IDs, person names, product names, dollar amounts, email id., preferences, etc).\n"
                    'If none found: {"entities": {}}'
                )
            ),
            HumanMessage(content=text),
        ]
        response = llm.invoke(extract_messages)
        try:
            import json

            data = json.loads(response.content)
            self.entities.update(data.get("entities", {}))
        except Exception:
            # In real production code, log this.
            pass

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self.extract_entities(user_text)
        self.messages.append(HumanMessage(content=user_text))
        self.messages.append(AIMessage(content=assistant_text))

    def get_context(self) -> list:
        context = [SystemMessage(content="You are a helpful assistant.")]
        if self.entities:
            entity_str = "\n".join([f"- {k}: {v}" for k, v in self.entities.items()])
            context.append(
                SystemMessage(
                    content=f"Known entities in this conversation:\n{entity_str}"
                )
            )
        context.extend(self.messages[-4:])  # last 2 exchanges
        return context

    def stats(self) -> dict:
        return {
            "entities_tracked": len(self.entities),
            "entity_keys": list(self.entities.keys()),
        }


if __name__ == "__main__":
    buffer_mem = BufferMemory()
    sliding_mem = SlidingWindowMemory(window_size=4)
    entity_mem = EntityMemory()

    print("=" * 60)
    print("DEMO 22: Interactive Memory Types Comparison")
    print("=" * 60)
    print("Type a message and press Enter to send it to the agent.")
    print("Commands:")
    print("  /stats  → show current memory stats")
    print("  /exit   → quit")
    print()

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("Exiting demo.")
            break

        if user_text.lower() in {"/stats", "stats"}:
            print("\n[Buffer]  stats:", buffer_mem.stats())
            print("[Sliding] stats:", sliding_mem.stats())
            print("[Entity]  stats:", entity_mem.stats())
            print("[Entity]  entities:", entity_mem.entities)
            print()
            continue

        print("\n--- Buffer Memory ---")
        buffer_mem.add_user(user_text)
        resp_buffer = llm.invoke(buffer_mem.get_context())
        buffer_mem.add_assistant(resp_buffer.content)
        print(resp_buffer.content)
        print("  [stats:", buffer_mem.stats(), "]")

        print("\n--- Sliding Window Memory ---")
        sliding_mem.add_user(user_text)
        resp_sliding = llm.invoke(sliding_mem.get_context())
        sliding_mem.add_assistant(resp_sliding.content)
        print(resp_sliding.content)
        print("  [stats:", sliding_mem.stats(), "]")

        print("\n--- Entity Memory ---")
        # For entity memory we call with current context and add the exchange
        resp_entity = llm.invoke(
            entity_mem.get_context() + [HumanMessage(content=user_text)]
        )
        entity_mem.add_exchange(user_text, resp_entity.content)
        print(resp_entity.content)
        print("  [stats:", entity_mem.stats(), "]")
        print("  [entities:", entity_mem.entities, "]")
        print("\n" + "=" * 60 + "\n")


