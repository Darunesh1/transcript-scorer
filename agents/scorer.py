import json
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# Define exact response structure using Pydantic
class CriterionDetails(BaseModel):
    """Details for each scoring criterion"""

    keywords_found: List[str] = Field(
        default_factory=list, description="Keywords found in transcript"
    )
    calculated_value: float = Field(description="Calculated metric value")
    reasoning: str = Field(description="Brief explanation of scoring")


class PerCriterion(BaseModel):
    """Individual criterion scoring result"""

    criterion: str = Field(description="Criterion name from rubric")
    metric: str = Field(description="Metric name from rubric")
    score: float = Field(description="Score earned")
    max_score: float = Field(description="Maximum possible score")
    feedback: str = Field(description="Specific feedback with examples")
    details: CriterionDetails


class ScoringResponse(BaseModel):
    """Complete scoring response structure"""

    overall_score: float = Field(description="Overall score 0-100")
    word_count: int = Field(description="Total word count")
    per_criterion: List[PerCriterion] = Field(description="Scores for each criterion")


class ScoringAgent:
    """
    Scorer agent using response schema for guaranteed valid JSON.
    Eliminates all JSON parsing errors!
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-2.5-flash-lite"
        print("  ✓ Scorer Agent initialized with response schema", flush=True)

    def score(self, transcript: str, rubric: dict, duration: int = None) -> dict:
        """
        Score transcript using enforced JSON schema.
        Response schema guarantees valid JSON - no more parsing errors!
        """

        prompt = f"""You are an expert transcript scoring agent. Score this transcript using the provided rubric.

TRANSCRIPT:
{transcript}

DURATION: {duration if duration else "Not provided"} seconds

RUBRIC:
{json.dumps(rubric, indent=2)}

SCORING INSTRUCTIONS:
1. Count words precisely from the transcript
2. Check for keywords (case-insensitive, partial match OK)
3. Calculate metrics accurately:
   - WPM = word_count / (duration/60) if duration provided
   - TTR = unique_words / total_words
   - Filler rate = (filler_count / total_words) * 100
   - Grammar errors, sentiment analysis
4. Apply rubric scoring rules exactly as specified
5. Sum all weighted scores for overall_score (0-100)
6. Provide specific feedback with examples from the transcript
7. Keep feedback concise (under 150 characters per criterion)

OUTPUT REQUIREMENTS:
- overall_score: Sum of all weighted criterion scores (0-100)
- word_count: Exact count of words in transcript
- per_criterion: Array with one entry per rubric criterion
  - criterion: Exact criterion name from rubric
  - metric: Exact metric name from rubric
  - score: Calculated score for this criterion
  - max_score: Maximum possible score from rubric
  - feedback: Specific explanation with examples (single line, under 150 chars)
  - details:
    - keywords_found: List of keywords detected
    - calculated_value: Numeric metric value
    - reasoning: Brief calculation explanation (single line)

Be precise with calculations and provide actionable feedback."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=ScoringResponse,  # ENFORCES EXACT STRUCTURE!
                ),
            )

            # Parse response - guaranteed to be valid JSON matching schema!
            result = json.loads(response.text)

            # Quick validation (should always pass with schema)
            required = ["overall_score", "word_count", "per_criterion"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            print(f"    ✓ Scoring successful", flush=True)
            return result

        except json.JSONDecodeError as e:
            # Should never happen with response_schema, but handle just in case
            print(f"    ❌ JSON parse error (unexpected with schema): {e}", flush=True)
            print(f"    Response was: {response.text[:500]}...", flush=True)
            raise ValueError(f"JSON parsing failed despite schema: {e}")

        except Exception as e:
            print(f"    ❌ Scoring error: {e}", flush=True)
            raise
