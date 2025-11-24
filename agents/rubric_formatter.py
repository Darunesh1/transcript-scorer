import json

from google import genai
from google.genai import types


class RubricFormatterAgent:
    """
    Specialized agent for formatting rubrics.
    ADK Pattern: Single-responsibility agent with tool-like behavior.
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-1.5-pro"
        print("  ✓ RubricFormatter Agent initialized")

    def format_rubric(self, raw_rubric_data: str) -> dict:
        """
        Format raw rubric into structured JSON.
        Uses JSON mode for reliable parsing.
        """

        system_instruction = """You are a rubric formatting specialist. Your job is to convert raw rubric data into clean, structured JSON.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Follow the exact structure specified
3. Extract all criteria, metrics, weights, keywords, and scoring rules
4. Preserve all information from the raw data

Output JSON structure:
{
  "criteria": [
    {
      "name": "Content & Structure",
      "total_weight": 40,
      "metrics": [
        {
          "metric_name": "Salutation Level",
          "weight": 5,
          "scoring_rules": [...],
          "keywords": [...]
        }
      ]
    }
  ]
}"""

        prompt = f"""Raw Rubric Data:
{raw_rubric_data}

Convert this into the structured JSON format. Include ALL information."""

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
                    temperature=0.1,
                    response_mime_type="application/json",  # Force JSON output
                ),
            )

            # Parse and validate JSON
            result = json.loads(response.text)

            if "criteria" not in result:
                raise ValueError("Formatted rubric missing 'criteria' field")

            return result

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}")
            print(f"    Response was: {response.text[:200]}...")
            raise
        except Exception as e:
            print(f"    ❌ Formatting error: {e}")
            raise
