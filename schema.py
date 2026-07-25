from pydantic import BaseModel, Field

class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A math word problem to solve.")
    max_new_tokens: int = Field(
        default=400, ge=16, le=1024,
        description="Max tokens to generate for the answer."
    )
 
 
class SolveResponse(BaseModel):
    question: str
    reasoning: str
    final_answer: str | None
    latency_seconds: float