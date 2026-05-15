from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectFeatures(BaseModel):
    project_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tech_stack: List[str] = Field(default_factory=list)
    installation_commands: List[str] = Field(default_factory=list)
    has_tests: bool = Field(default=False)
    license_type: Optional[str] = Field(default=None)


class UserIntent(BaseModel):
    main_problem: str = ""
    expected_behavior: str = ""
    edge_cases: str = ""


class ConfidenceScores(BaseModel):
    project_name: float = 0.0
    description: float = 0.0
    tech_stack: float = 0.0
    installation_commands: float = 0.0
    has_tests: float = 0.0
    license_type: float = 0.0


class TestScenario(BaseModel):
    name: str
    type: str  # positive | negative | edge | boundary
    description: str
    expected_result: str
    priority: str = "medium"  # high | medium | low


class FinalOutput(BaseModel):
    features: ProjectFeatures
    user_intent: UserIntent
    test_scenarios: List[TestScenario] = Field(default_factory=list)
    confidence: ConfidenceScores = Field(default_factory=ConfidenceScores)
    low_confidence_fields: List[str] = Field(default_factory=list)

    def to_api_dict(self) -> dict:
        return {
            "features": self.features.model_dump(),
            "confidence": self.confidence.model_dump(),
            "low_confidence_fields": self.low_confidence_fields,
            "test_scenarios": [s.model_dump() for s in self.test_scenarios],
        }
