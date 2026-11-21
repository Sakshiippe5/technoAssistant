
from pydantic import BaseModel, Field


class TroubleshootRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=2000
    )


class TroubleshootResponse(BaseModel):

    query: str
    status: str
    message: str | None = None

    total_issues: int = 0
    supported_issues: int = 0

    issues: list[dict] = Field(
        default_factory=list
    )

    latency_seconds: float
