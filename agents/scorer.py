import json
import re

from google import genai
from google.genai import types


class ScoringAgent:
    """
    Scorer agent with robust JSON parsing and error recovery.
    Handles malformed JSON from AI responses.
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-2.5-flash-lite"
        print("  ✓ Scorer Agent initialized", flush=True)

    def score(self, transcript: str, rubric: dict, duration: int = None) -> dict:
        """
        Score transcript with JSON cleaning and error recovery.
        Uses strict prompt to prevent malformed JSON.
        """

        system_instruction = """You are an expert transcript scoring agent.

CRITICAL JSON RULES (MUST FOLLOW):
1. Return ONLY valid JSON (no markdown, no code blocks, no backticks)
2. ALL feedback must be SINGLE LINE (no newlines, no line breaks)
3. Escape ALL quotes inside strings using backslash
4. Keep feedback under 150 characters
5. Use simple punctuation only (period, comma)

Scoring Process:
1. Count words precisely from the transcript
2. Check for keywords (case-insensitive, partial match OK)
3. Calculate metrics accurately (WPM, TTR, filler rate, etc.)
4. Apply rubric scoring rules exactly as specified
5. Sum all weighted scores for overall_score (0-100)

Output JSON (EXACTLY THIS FORMAT):
{
  "overall_score": 85.5,
  "word_count": 120,
  "per_criterion": [
    {
      "criterion": "Content & Structure",
      "metric": "Salutation Level",
      "score": 4.0,
      "max_score": 5.0,
      "feedback": "Good greeting detected. Found keywords: name, age, school. Clear introduction structure.",
      "details": {
        "keywords_found": ["name", "age", "school"],
        "calculated_value": 4.0,
        "reasoning": "Strong salutation with good keyword coverage"
      }
    }
  ]
}

REMEMBER: 
- Single line feedback only (no newlines)
- Escape quotes inside strings
- No markdown formatting
- Valid JSON only"""

        prompt = f"""TRANSCRIPT:
{transcript}

DURATION: {duration if duration else "Not provided"} seconds

RUBRIC:
{json.dumps(rubric, indent=2)}

Score this transcript precisely following all rubric criteria. Return ONLY valid JSON."""

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
                    response_mime_type="application/json",
                ),
            )

            # Get response text
            response_text = response.text.strip()

            # Clean response (remove markdown if present)
            if response_text.startswith("```"):
                response_text = re.sub(r"```json\s*|\s*```", "", response_text)

            # Basic JSON cleaning
            response_text = self._clean_json(response_text)

            # Try to parse JSON
            result = json.loads(response_text)

            # Validate required fields
            required = ["overall_score", "word_count", "per_criterion"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            print(f"    ✓ Scoring successful", flush=True)
            return result

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}", flush=True)
            print(f"    Response was: {response.text[:500]}...", flush=True)

            # Try aggressive cleaning as last resort
            try:
                print(f"    → Attempting aggressive JSON recovery...", flush=True)
                cleaned = self._aggressive_json_clean(response.text)
                result = json.loads(cleaned)

                # Validate after recovery
                required = ["overall_score", "word_count", "per_criterion"]
                for field in required:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")

                print(f"    ✓ Recovered with aggressive cleaning", flush=True)
                return result
            except Exception as recovery_error:
                print(f"    ❌ Recovery failed: {recovery_error}", flush=True)
                raise ValueError(f"JSON parse failed after recovery attempts: {e}")

        except Exception as e:
            print(f"    ❌ Scoring error: {e}", flush=True)
            raise

    def _clean_json(self, text: str) -> str:
        """
        Clean common JSON issues.
        Removes text before first { and after last }.
        """
        # Remove any text before first {
        start = text.find("{")
        if start > 0:
            text = text[start:]

        # Remove any text after last }
        end = text.rfind("}")
        if end > 0:
            text = text[: end + 1]

        return text

    def _aggressive_json_clean(self, text: str) -> str:
        """
        Aggressive JSON cleaning for malformed responses.
        Fixes unescaped newlines inside string values.
        """
        # Extract JSON part
        text = self._clean_json(text)

        # Fix unescaped newlines in strings (most common issue)
        # Strategy: Track if we're inside a string and merge lines
        lines = text.split("\n")
        fixed_lines = []
        in_string = False

        for line in lines:
            # Count unescaped quotes to track string state
            quote_count = line.count('"') - line.count('\\"')

            if in_string:
                # We're inside a string, merge this line with previous
                if fixed_lines:
                    # Replace newline with space
                    fixed_lines[-1] += " " + line.strip()
            else:
                # Normal line
                fixed_lines.append(line)

            # Toggle string state if odd number of quotes
            if quote_count % 2 == 1:
                in_string = not in_string

        cleaned = "\n".join(fixed_lines)

        # Additional fixes
        # Fix common escaping issues
        cleaned = cleaned.replace('\n"', '"')  # Remove newlines before quotes
        cleaned = cleaned.replace('"\n', '"')  # Remove newlines after quotes

        return cleaned
