
from pydantic import BaseModel, Field


class TroubleshootingStep(BaseModel):
    action: str = Field(min_length=1)
    source_id: str


class Resolution(BaseModel):
    condition: str
    action: str
    source_id: str


class Escalation(BaseModel):
    team: str
    condition: str
    source_id: str


class TroubleshootingResponse(BaseModel):
    issue: str
    summary: str
    possible_causes: list[TroubleshootingStep]
    diagnostic_steps: list[TroubleshootingStep]
    resolutions: list[Resolution]
    escalation: list[Escalation]
    required_information: list[TroubleshootingStep]
    source_ids: list[str] = Field(min_length=1)


def validate_response(response, documents):

    allowed_sources = {
        doc["chunk_id"]: doc
        for doc in documents
    }

    cited_sources = set(response.source_ids)

    if not cited_sources.issubset(allowed_sources):
        raise ValueError("Invalid source IDs generated.")

    sections = [
        response.possible_causes,
        response.diagnostic_steps,
        response.resolutions,
        response.escalation,
        response.required_information
    ]

    for section in sections:

        for item in section:

            if item.source_id not in cited_sources:
                raise ValueError(
                    f"Invalid citation: {item.source_id}"
                )

    for escalation in response.escalation:

        metadata = allowed_sources[
            escalation.source_id
        ]["metadata"]

        expected_team = metadata["escalation_team"]

        if escalation.team.strip().casefold() != (
            expected_team.strip().casefold()
        ):
            raise ValueError(
                "Unsupported escalation team."
            )

    if not response.diagnostic_steps:
        raise ValueError(
            "No diagnostic steps generated."
        )

    return response.model_dump()
