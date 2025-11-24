import os
import sys
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import OrchestratorAgent

app = FastAPI(
    title="Transcript Scorer API",
    description="AI-powered transcript scoring using Google Gemini ADK",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Initialize orchestrator with retry support
print("🚀 Starting API with ADK Orchestrator...")
orchestrator = OrchestratorAgent()
print("✓ API ready\n")


@app.get("/")
async def root():
    return {
        "message": "Transcript Scorer API (ADK Pattern)",
        "status": "running",
        "version": "1.0.0",
        "features": ["retry_logic", "multi_agent", "error_handling"],
    }


@app.post("/score", response_model=ScoringResponse)
async def score_transcript(
    transcript: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None),
    rubric_file: Optional[UploadFile] = File(None),
    duration_seconds: Optional[int] = Form(None),
):
    """
    Score transcript with ADK multi-agent orchestration and retry logic.
    """

    try:
        # Validate input
        if not transcript and not transcript_file:
            raise HTTPException(
                status_code=400,
                detail="Must provide either transcript text or transcript file",
            )

        # Prepare inputs
        transcript_input = transcript if transcript else transcript_file
        rubric_input = rubric_file if rubric_file else None

        # Call orchestrator with retry logic
        result = orchestrator.process(
            transcript_input=transcript_input,
            rubric_input=rubric_input,
            duration=duration_seconds,
            max_retries=3,  # Configurable retry attempts
        )

        return ScoringResponse(**result)

    except ValueError as e:
        # User input errors or validation failures
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        # File loading errors
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        # Unexpected errors
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "orchestrator": "initialized",
        "agents": ["formatter", "scorer"],
        "retry_enabled": True,
    }
