# Error Handling across agents
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional


# ==================== RETRY LOGIC ====================

class RetryDemo:
    """Demonstrates retry logic with exponential backoff"""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.attempts = []

    def call_with_retry(self, func: Callable, *args, **kwargs):
        """Call function with retry logic"""
        print(f"\n{'=' * 70}")
        print("🔄 RETRY LOGIC DEMONSTRATION")
        print(f"{'=' * 70}\n")

        for attempt in range(1, self.max_attempts + 1):
            print(f"Attempt {attempt}/{self.max_attempts}:")

            try:
                result = func(*args, **kwargs)
                print(f"   ✅ Success on attempt {attempt}!")
                return result
            except Exception as e:
                self.attempts.append({
                    'attempt': attempt,
                    'error': str(e),
                    'timestamp': time.time()
                })

                if attempt < self.max_attempts:
                    wait_time = 2 ** (attempt - 1)  # Exponential: 1s, 2s, 4s
                    print(f"   ❌ Failed: {e}")
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed after {self.max_attempts} attempts")
                    raise

        raise Exception("All retries exhausted")


# ==================== FALLBACK MECHANISM ====================

class FallbackDemo:
    """Demonstrates fallback mechanism"""

    def __init__(self):
        self.primary_called = False
        self.fallback_called = False

    def primary_handler(self, ticket: str) -> str:
        """Primary specialist (simulates failure)"""
        self.primary_called = True
        print("   🔴 Primary specialist called...")

        # Simulate failure
        if random.random() < 0.7:  # 70% failure rate for demo
            raise Exception("Primary specialist unavailable")

        return "Primary response"

    def fallback_handler(self, ticket: str) -> str:
        """Fallback specialist"""
        self.fallback_called = True
        print("   🟡 Fallback specialist called...")
        return "Fallback response (general specialist)"

    def handle_with_fallback(self, ticket: str) -> str:
        """Handle with fallback mechanism"""
        print(f"\n{'=' * 70}")
        print("🛡️  FALLBACK MECHANISM DEMONSTRATION")
        print(f"{'=' * 70}\n")
        print(f"Ticket: {ticket}\n")

        try:
            print("Step 1: Try primary specialist...")
            response = self.primary_handler(ticket)
            print("   ✅ Primary specialist succeeded!")
            return response
        except Exception as e:
            print(f"   ❌ Primary specialist failed: {e}")
            print("\nStep 2: Fall back to general specialist...")
            response = self.fallback_handler(ticket)
            print("   ✅ Fallback specialist succeeded!")
            return response


# ==================== CIRCUIT BREAKER ====================

class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, don't call
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerDemo:
    """Demonstrates circuit breaker pattern"""

    max_failures: int = 3
    timeout: int = 60
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[float] = None
    half_open_attempts: int = 0

    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker"""
        print(f"\n{'=' * 70}")
        print("⚡ CIRCUIT BREAKER DEMONSTRATION")
        print(f"{'=' * 70}\n")

        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                print("   🔄 Timeout expired, moving to HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
            else:
                print(f"   🔴 Circuit is OPEN - failing fast")
                print(f"   ⏳ Wait {self.timeout - int(time.time() - self.last_failure_time)}s")
                raise Exception("Circuit breaker is OPEN")

        # Try the call
        try:
            result = func(*args, **kwargs)

            # Success!
            if self.state == CircuitState.HALF_OPEN:
                print("   ✅ Half-open test succeeded, closing circuit")
                self.state = CircuitState.CLOSED
                self.failures = 0
            elif self.state == CircuitState.CLOSED:
                print("   ✅ Call succeeded, circuit remains CLOSED")
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            print(f"   ❌ Call failed (failure {self.failures}/{self.max_failures})")

            if self.failures >= self.max_failures:
                print(f"   🔴 Max failures reached, opening circuit")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                print(f"   🔴 Half-open test failed, reopening circuit")
                self.state = CircuitState.OPEN

            raise


# ==================== GRACEFUL DEGRADATION ====================

class GracefulDegradationDemo:
    """Demonstrates graceful degradation"""

    def __init__(self):
        self.layers = []

    def layer_1_retry(self, func: Callable, *args, **kwargs):
        """Layer 1: Retry with exponential backoff"""
        print("\n" + "=" * 70)
        print("🛡️  GRACEFUL DEGRADATION - 3 LAYERS")
        print("=" * 70 + "\n")

        print("Layer 1: Retry with exponential backoff")
        retry = RetryDemo(max_attempts=3)

        try:
            return retry.call_with_retry(func, *args, **kwargs)
        except Exception as e:
            print(f"   ❌ Layer 1 failed: {e}")
            self.layers.append('layer_1_failed')
            raise

    def layer_2_fallback(self, primary_func: Callable, fallback_func: Callable, *args, **kwargs):
        """Layer 2: Fallback to general specialist"""
        print("\nLayer 2: Fallback to general specialist")

        try:
            return self.layer_1_retry(primary_func, *args, **kwargs)
        except Exception:
            try:
                return fallback_func(*args, **kwargs)
            except Exception as e:
                print(f"   ❌ Layer 2 failed: {e}")
                self.layers.append('layer_2_failed')
                raise

    def layer_3_escalate(self, primary_func: Callable, fallback_func: Callable, *args, **kwargs):
        """Layer 3: Escalate to human"""
        print("\nLayer 3: Escalate to human support")

        try:
            return self.layer_2_fallback(primary_func, fallback_func, *args, **kwargs)
        except Exception:
            print("   🚨 All layers failed, escalating to human")
            self.layers.append('layer_3_escalated')
            return "I've escalated your request to our support team. You will receive a response within 2 hours."


# ==================== DEMO FUNCTIONS ====================

def simulate_unreliable_service(ticket: str) -> str:
    """Simulate an unreliable service (fails 70% of the time)"""
    if random.random() < 0.7:
        raise Exception("Service temporarily unavailable")
    return f"Response to: {ticket}"


def simulate_reliable_fallback(ticket: str) -> str:
    """Simulate a reliable fallback service"""
    return f"General response to: {ticket}"


# ==================== MAIN DEMO ====================

def main():
    """Run the error handling demo"""

    print("\n" + "=" * 70)
    print("⚠️  DEMO 4: Error Handling Across Agents")
    print("=" * 70)
    print("\nThis demo shows:")
    print("  1. Retry logic with exponential backoff")
    print("  2. Fallback mechanisms")
    print("  3. Circuit breaker pattern")
    print("  4. Graceful degradation (3 layers)")
    print("\n" + "-" * 70 + "\n")

    # Demo 1: Retry Logic
    print("\n" + "─" * 70)
    print("DEMO 1: RETRY LOGIC")
    print("─" * 70)

    retry_demo = RetryDemo(max_attempts=3)
    try:
        result = retry_demo.call_with_retry(simulate_unreliable_service, "Test ticket")
        print(f"\n✅ Final result: {result}")
    except Exception as e:
        print(f"\n❌ All retries exhausted: {e}")

    input("\nPress Enter to continue to Fallback demo...")

    # Demo 2: Fallback
    print("\n" + "─" * 70)
    print("DEMO 2: FALLBACK MECHANISM")
    print("─" * 70)

    fallback_demo = FallbackDemo()
    result = fallback_demo.handle_with_fallback("I need help with billing")
    print(f"\n✅ Final result: {result}")

    input("\nPress Enter to continue to Circuit Breaker demo...")

    # Demo 3: Circuit Breaker
    print("\n" + "─" * 70)
    print("DEMO 3: CIRCUIT BREAKER")
    print("─" * 70)

    circuit = CircuitBreakerDemo(max_failures=3, timeout=5)  # Short timeout for demo

    print("Simulating 5 calls to failing service...\n")
    for i in range(5):
        print(f"\nCall {i + 1}:")
        try:
            circuit.call(simulate_unreliable_service, "Test ticket")
        except Exception as e:
            print(f"   Error: {e}")
        time.sleep(0.5)

    print(f"\n📊 Circuit State: {circuit.state.value}")
    print(f"   Failures: {circuit.failures}")

    input("\nPress Enter to continue to Graceful Degradation demo...")

    # Demo 4: Graceful Degradation
    print("\n" + "─" * 70)
    print("DEMO 4: GRACEFUL DEGRADATION (3 LAYERS)")
    print("─" * 70)

    degradation = GracefulDegradationDemo()
    result = degradation.layer_3_escalate(
        simulate_unreliable_service,
        simulate_reliable_fallback,
        "I need urgent help"
    )

    print(f"\n✅ Final result: {result}")

    # Summary
    print("\n" + "=" * 70)
    print("📋 ERROR HANDLING SUMMARY")
    print("=" * 70 + "\n")

    print("1. RETRY LOGIC:")
    print("   • Handles transient failures")
    print("   • Exponential backoff prevents hammering")
    print("   • Max attempts prevent infinite retries")

    print("\n2. FALLBACK MECHANISM:")
    print("   • Primary specialist fails → fallback to general")
    print("   • Prevents total system failure")
    print("   • Maintains service availability")

    print("\n3. CIRCUIT BREAKER:")
    print("   • Prevents cascading failures")
    print("   • Fails fast when service is down")
    print("   • Automatic recovery testing")

    print("\n4. GRACEFUL DEGRADATION:")
    print("   • Layer 1: Retry")
    print("   • Layer 2: Fallback")
    print("   • Layer 3: Escalate to human")
    print("   • Never fail silently!")

    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70 + "\n")

    print("💡 Key Takeaways:")
    print("  • Always have multiple layers of defense")
    print("  • Retry transient failures")
    print("  • Fallback to general specialist")
    print("  • Escalate to human as last resort")
    print("  • Never fail silently\n")


if __name__ == "__main__":
    main()