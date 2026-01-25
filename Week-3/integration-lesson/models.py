"""
Pydantic models for structured output.
This ensures type-safe responses from the LLM.
"""

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """Structured response from the customer support agent."""
    
    reasoning: str = Field(
        description="Your step-by-step reasoning for this decision"
    )
    action: str = Field(
        description="Action to take: process_refund|escalate_to_human|provide_info|create_approval_ticket"
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    message: str = Field(
        description="Message to send to the user"
    )
    requires_approval: bool = Field(
        description="Does this action require human approval?"
    )
