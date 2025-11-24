# agents/scorer.py - ADK Way (Kaggle Pattern)
import json

from google import genai
from google.genai import types


class ScoringAgent:
    """
    Scorer using ADK SDK (google-genai).
    Simple approach: AI does calculations (no tools needed).
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-2.5-flash-lite"
        print("  ✓ Scorer Agent initialized (ADK)")

    def score(
        self,
        transcript: str,
        rubric: dict,
        duration: int = None,
        max_correction_attempts: int = 2,
    ) -> dict:
        """Score with retry"""
        for attempt in range(1, max_correction_attempts + 1):
            try:
                print(f"    → Scoring (attempt {attempt}/{max_correction_attempts})")

                result = self._score_simple(transcript, rubric, duration)
                self._validate_result(result)

                print(f"    ✓ Valid result")
                return result

            except Exception as e:
                print(f"    ⚠️  Error: {str(e)[:100]}")
                if attempt == max_correction_attempts:
                    raise ValueError(f"Failed: {e}")

        raise ValueError("Max attempts")

    def _score_simple(
        self, transcript: str, rubric: dict, duration: int = None
    ) -> dict:
        """
        Simple scoring without tools (ADK pattern).
        AI does all calculations internally.
        """

        system_instruction = """You are an expert transcript scoring agent.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Count words precisely from the transcript
3. Check for keywords (case-insensitive, partial match OK)
4. Calculate metrics accurately:
   - WPM = word_count / (duration/60) if duration provided
   - TTR = unique_words / total_words
   - Filler rate = (filler_count / total_words) * 100
   - Grammar errors, sentiment analysis
5. Apply rubric scoring rules exactly
6. Sum all weighted scores for overall_score (0-100)

Provide specific feedback with examples from the transcript.

Output JSON:
{
  "overall_score": <0-100>,
  "word_count": <integer>,
  "per_criterion": [
    {
      "criterion": "<criterion name from rubric>",
      "metric": "<metric name from rubric>",
      "score": <float>,
      "max_score": <float>,
      "feedback": "<specific explanation with examples>",
      "details": {
        "keywords_found": ["list"],
        "calculated_value": <number>,
        "reasoning": "<brief explanation>"
      }
    }
  ]
}"""

        prompt = f"""TRANSCRIPT:
{transcript}

DURATION: {duration if duration else "Not provided"} seconds

RUBRIC:
{json.dumps(rubric, indent=2)}

Score this transcript precisely following all rubric criteria. Show your calculations."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=system_instruction + "\n\n" + prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2, response_mime_type="application/json"
                ),
            )

            # Parse JSON
            result = json.loads(response.text)

            # Validate required fields
            required = ["overall_score", "word_count", "per_criterion"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing: {field}")

            return result

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON error: {e}")
            if response and response.text:
                print(f"    Response: {response.text[:200]}...")
            raise
        except Exception as e:
            print(f"    ❌ Error: {e}")
            raise

    def _validate_result(self, result: dict):
        """Validate result"""
        if "overall_score" not in result or "per_criterion" not in result:
            raise ValueError("Missing required fields")

        if not (0 <= result["overall_score"] <= 100):
            raise ValueError(f"Invalid score: {result['overall_score']}")

        if not result["per_criterion"]:
            raise ValueError("Empty per_criterion")
