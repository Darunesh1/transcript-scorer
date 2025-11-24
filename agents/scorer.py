import json

from google import genai
from google.genai import types


class ScoringAgent:
    """
    Specialized agent for scoring transcripts.
    ADK Pattern: Task-specific agent with structured output.
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-2.5-flash-lite"
        print("  ✓ Scorer Agent initialized")

    def score(self, transcript: str, rubric: dict, duration: int = None) -> dict:
        """
        Score transcript against rubric with detailed feedback.
        Uses JSON mode for reliable structured output.
        """

        system_instruction = """You are an expert transcript scoring agent. You evaluate transcripts against rubrics with precision and provide detailed feedback.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Be precise with calculations (count words, check keywords, calculate metrics)
3. Provide specific feedback with examples from the transcript
4. Ensure all scores are within valid ranges
5. Sum all weighted scores correctly for overall_score (0-100)

Scoring Process:
1. Count total words in transcript
2. For each criterion:
   - Check for keywords (case-insensitive, partial match OK)
   - Calculate metrics (WPM, TTR, filler rate, etc.)
   - Apply scoring rules from rubric
   - Provide actionable feedback with evidence
3. Calculate overall_score by summing all weighted scores

Output JSON structure:
{
  "overall_score": <0-100>,
  "word_count": <integer>,
  "per_criterion": [
    {
      "criterion": "<name>",
      "metric": "<metric>",
      "score": <earned>,
      "max_score": <possible>,
      "feedback": "<specific explanation with examples>",
      "details": {
        "keywords_found": ["list"],
        "calculated_value": <number>,
        "reasoning": "<calculation explanation>"
      }
    }
  ]
}"""

        prompt = f"""TRANSCRIPT:
{transcript}

DURATION: {duration if duration else "Not provided"} seconds

RUBRIC:
{json.dumps(rubric, indent=2)}

Score this transcript precisely following all rubric criteria."""

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
                    temperature=0.2,
                    response_mime_type="application/json",  # Force JSON output
                ),
            )

            # Parse and validate JSON
            result = json.loads(response.text)

            # Validate required fields
            required = ["overall_score", "word_count", "per_criterion"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            return result

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}")
            print(f"    Response was: {response.text[:200]}...")
            raise
        except Exception as e:
            print(f"    ❌ Scoring error: {e}")
            raise
