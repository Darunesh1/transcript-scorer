from typing import List

from pydantic import BaseModel


class CriterionResult(BaseModel):
    criterion: str
    metric: str
    score: float
    max_score: float
    feedback: str
    details: dict


class ScoringResponse(BaseModel):
    overall_score: float
    max_overall_score: float = 100
    word_count: int
    per_criterion: List[CriterionResult]
