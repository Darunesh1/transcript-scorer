import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agents.rubric_formatter import RubricFormatterAgent
from agents.scorer import ScoringAgent
from utils.file_parser import extract_text_from_file
from utils.rubric_loader import load_default_rubric, parse_uploaded_rubric

load_dotenv()


class OrchestratorAgent:
    """
    ADK Multi-Agent Orchestrator (Kaggle Day 5 pattern).
    Coordinates RubricFormatter and Scorer agents with two-level retry:
    - Inner: AI self-correction (scorer handles this)
    - Outer: Complete orchestrator retry with exponential backoff
    """

    def __init__(self):
        print("🚀 Initializing Orchestrator Agent (ADK Pattern)...")

        # Initialize Gemini client with retry configuration
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=1.0,
                    max_delay=10.0,
                    exp_base=2.0,
                    jitter=0.5,
                    http_status_codes=[
                        429,
                        503,
                        500,
                    ],  # Rate limit, unavailable, server error
                )
            ),
        )

        # Initialize sub-agents (lazy load formatter)
        self.formatter_agent = None
        self.scorer_agent = ScoringAgent(self.client)

        print("✓ Orchestrator initialized with two-level retry")
        print("✓ Scorer agent ready with tool calling + self-correction")

    def _get_formatter_agent(self):
        """Lazy load formatter agent (only for custom rubrics)"""
        if self.formatter_agent is None:
            print("  → Initializing RubricFormatter agent...")
            self.formatter_agent = RubricFormatterAgent(self.client)
        return self.formatter_agent

    def process(
        self, transcript_input, rubric_input=None, duration=None, max_retries=3
    ):
        """
        Main orchestration workflow (ADK Day 5 pattern).

        Steps:
        1. Extract transcript
        2. Load/format rubric (skip formatter for default)
        3. Score with tool-calling agent (two-level retry)
        4. Validate results
        """
        try:
            print("\n" + "=" * 60)
            print("🎯 ORCHESTRATION START (ADK Multi-Agent)")
            print("=" * 60)

            # Step 1: Extract transcript
            print("\n📝 Step 1: Extracting transcript...")
            transcript = self._extract_transcript(transcript_input)
            word_count = len(transcript.split())
            print(f"✓ Transcript: {len(transcript)} chars, {word_count} words")

            # Step 2: Load and format rubric (smart loading)
            print("\n📋 Step 2: Loading rubric...")
            formatted_rubric = self._load_and_format_rubric(rubric_input, max_retries)
            criteria_count = len(formatted_rubric.get("criteria", []))
            print(f"✓ Rubric ready: {criteria_count} criteria")

            # Step 3: Score with tool-calling agent (two-level retry)
            print("\n🎯 Step 3: Scoring with tool-calling agent (two-level retry)...")
            results = self._score_with_retry(
                transcript, formatted_rubric, duration, max_retries
            )
            print(f"✓ Scoring complete: {results['overall_score']:.1f}/100")

            # Step 4: Validate results
            print("\n✅ Step 4: Validating results...")
            self._validate_results(results)
            print("✓ Validation passed")

            print("\n" + "=" * 60)
            print("✨ ORCHESTRATION COMPLETE")
            print("=" * 60 + "\n")

            return results

        except ValueError as e:
            # User-facing errors (bad input, invalid format)
            print(f"\n⚠️  User Error: {str(e)}")
            raise
        except Exception as e:
            # System errors
            print(f"\n❌ System Error: {str(e)}")
            raise

    def _extract_transcript(self, transcript_input):
        """Extract transcript from string or file"""
        if isinstance(transcript_input, str):
            transcript = transcript_input.strip()
        else:
            transcript = extract_text_from_file(transcript_input)

        if not transcript:
            raise ValueError("Transcript is empty")

        return transcript

    def _load_and_format_rubric(self, rubric_input, max_retries):
        """
        Smart rubric loading (ADK best practice):
        - Default (None) → Load pre-formatted JSON, skip agent
        - Custom JSON (formatted) → Use directly, skip agent
        - Custom Excel/unformatted → Use formatter agent with retry
        """

        if rubric_input is None:
            # Default rubric - pre-formatted
            print("  → Using default rubric (pre-formatted JSON)")
            rubric = load_default_rubric()

            if isinstance(rubric, dict) and "criteria" in rubric:
                print("  ✓ Default rubric loaded (formatter agent NOT called)")
                return rubric
            else:
                raise ValueError("Default rubric is malformed")

        else:
            # Custom uploaded rubric
            print("  → Custom rubric uploaded")
            raw_rubric = parse_uploaded_rubric(rubric_input)

            # Check if already formatted (JSON with criteria)
            if isinstance(raw_rubric, dict) and "criteria" in raw_rubric:
                print("  ✓ Rubric already formatted (JSON)")
                return raw_rubric

            # Needs formatting (Excel or unstructured)
            print("  → Calling RubricFormatter agent...")
            return self._format_rubric_with_retry(raw_rubric, max_retries)

    def _format_rubric_with_retry(self, raw_rubric, max_retries=3):
        """
        Format rubric with exponential backoff retry (ADK Day 4 pattern).
        Retries on JSON parse errors or invalid structure.
        """
        formatter = self._get_formatter_agent()

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Formatting attempt {attempt}/{max_retries}")

                formatted = formatter.format_rubric(str(raw_rubric))

                # Validate structure
                if not isinstance(formatted, dict):
                    raise ValueError(f"Invalid type: {type(formatted)}")

                if "criteria" not in formatted:
                    raise ValueError("Missing 'criteria' field")

                if not isinstance(formatted["criteria"], list):
                    raise ValueError("'criteria' must be a list")

                if len(formatted["criteria"]) == 0:
                    raise ValueError("'criteria' list is empty")

                print(f"  ✓ Format valid on attempt {attempt}")
                return formatted

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  ⚠️  Format error: {e}")

                if attempt == max_retries:
                    raise ValueError(
                        f"Failed to format rubric after {max_retries} attempts. "
                        f"Please check your rubric file format. Error: {e}"
                    )

                # Exponential backoff
                wait_time = 2**attempt
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                raise

        raise ValueError("Max retries reached")

    def _score_with_retry(self, transcript, rubric, duration, max_retries=3):
        """
        Score with tool-calling agent with two-level retry (ADK Day 2 + Day 4).

        OUTER RETRY (Orchestrator level):
        - Complete retry with fresh conversation
        - Exponential backoff between attempts
        - Up to max_retries attempts

        INNER RETRY (Scorer level):
        - AI self-correction with error feedback
        - Handled inside scorer.score()
        - Up to 2 correction attempts

        Total resilience: Up to 6 attempts (3 outer × 2 inner)
        """

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Orchestrator retry attempt {attempt}/{max_retries}")

                # Call scorer agent (it has its own self-correction retry)
                result = self.scorer_agent.score(
                    transcript=transcript,
                    rubric=rubric,
                    duration=duration,
                    max_correction_attempts=2,  # Inner: AI gets 2 tries to self-correct
                )

                # Validate response structure
                self._validate_scoring_response(result)

                print(f"  ✓ Scoring valid on orchestrator attempt {attempt}")
                return result

            except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
                print(f"  ⚠️  Error on orchestrator attempt {attempt}: {e}")

                if attempt == max_retries:
                    raise ValueError(
                        f"Scoring failed after {max_retries} orchestrator attempts. "
                        f"AI could not produce valid output. Last error: {e}"
                    )

                # Exponential backoff before retry
                wait_time = 2**attempt
                print(f"  ⏳ Orchestrator retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                raise

        raise ValueError("Max orchestrator retries reached")

    def _validate_scoring_response(self, result):
        """Validate scoring response structure (orchestrator-level validation)"""
        # Required top-level fields
        required_fields = ["overall_score", "word_count", "per_criterion"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: '{field}'")

        # Validate score range
        if not isinstance(result["overall_score"], (int, float)):
            raise ValueError(
                f"overall_score must be a number, got {type(result['overall_score'])}"
            )

        if not (0 <= result["overall_score"] <= 100):
            raise ValueError(
                f"overall_score must be 0-100, got {result['overall_score']}"
            )

        # Validate per_criterion structure
        if not isinstance(result["per_criterion"], list):
            raise ValueError("per_criterion must be a list")

        if len(result["per_criterion"]) == 0:
            raise ValueError("per_criterion list is empty")

        # Validate each criterion entry
        required_criterion_fields = [
            "criterion",
            "metric",
            "score",
            "max_score",
            "feedback",
            "details",
        ]
        for i, criterion in enumerate(result["per_criterion"]):
            for field in required_criterion_fields:
                if field not in criterion:
                    raise ValueError(f"Criterion {i} missing field: '{field}'")

            # Validate score ranges
            if not isinstance(criterion["score"], (int, float)):
                raise ValueError(f"Criterion {i} score must be numeric")

            if criterion["score"] < 0 or criterion["score"] > criterion["max_score"]:
                raise ValueError(
                    f"Criterion {i} score ({criterion['score']}) exceeds max ({criterion['max_score']})"
                )

    def _validate_results(self, results):
        """Final validation of complete results"""
        if not results:
            raise ValueError("Results are empty")

        # Already validated in _validate_scoring_response, but double-check
        if results["overall_score"] < 0 or results["overall_score"] > 100:
            raise ValueError(f"Invalid overall score: {results['overall_score']}")

        if results["word_count"] < 0:
            raise ValueError(f"Invalid word count: {results['word_count']}")
