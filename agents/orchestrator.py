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
    Orchestrator Agent using ADK patterns.
    Smart rubric loading: skips formatter for pre-formatted default rubric.
    """

    def __init__(self):
        print("🚀 Initializing Orchestrator Agent with ADK patterns...")

        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=1,
                    max_delay=10,
                    exp_base=2.0,
                    jitter=0.5,
                    http_status_codes=[429, 503, 500],
                )
            ),
        )

        # Initialize sub-agents (only created when needed)
        self.formatter_agent = None
        self.scorer_agent = ScoringAgent(self.client)

        print("✓ Orchestrator initialized with retry logic")
        print("✓ Scorer agent ready")

    def _get_formatter_agent(self):
        """Lazy load formatter agent (only when needed for custom rubrics)"""
        if self.formatter_agent is None:
            print("  → Initializing RubricFormatter agent...")
            self.formatter_agent = RubricFormatterAgent(self.client)
        return self.formatter_agent

    def process(
        self, transcript_input, rubric_input=None, duration=None, max_retries=3
    ):
        """
        Main orchestration with smart rubric handling.

        Workflow:
        1. Extract transcript text
        2. Load rubric:
           - If default → use pre-formatted JSON (skip Agent 1)
           - If uploaded → format with Agent 1 (with retry)
        3. Score transcript (Agent 2) with retry
        4. Validate and return results
        """
        try:
            print("\n" + "=" * 60)
            print("🎯 ORCHESTRATION START")
            print("=" * 60)

            # Step 1: Extract transcript
            print("\n📝 Step 1: Extracting transcript...")
            transcript = self._extract_transcript(transcript_input)
            print(
                f"✓ Transcript: {len(transcript)} chars, {len(transcript.split())} words"
            )

            # Step 2: Load rubric (smart loading)
            print("\n📋 Step 2: Loading rubric...")
            formatted_rubric = self._load_and_format_rubric(rubric_input, max_retries)
            print(
                f"✓ Rubric ready: {len(formatted_rubric.get('criteria', []))} criteria"
            )

            # Step 3: Score with retry (Agent 2)
            print("\n🎯 Step 3: Scoring transcript (Scorer Agent)...")
            results = self._score_with_retry(
                transcript, formatted_rubric, duration, max_retries
            )
            print(f"✓ Scoring complete: {results['overall_score']:.1f}/100")

            # Step 4: Validate final results
            print("\n✅ Step 4: Validating results...")
            self._validate_results(results)
            print("✓ Validation passed")

            print("\n" + "=" * 60)
            print("✨ ORCHESTRATION COMPLETE")
            print("=" * 60 + "\n")

            return results

        except Exception as e:
            print(f"\n❌ Orchestration failed: {str(e)}")
            raise

    def _extract_transcript(self, transcript_input):
        """Extract transcript from string or file."""
        if isinstance(transcript_input, str):
            return transcript_input.strip()
        else:
            return extract_text_from_file(transcript_input)

    def _load_and_format_rubric(self, rubric_input, max_retries: int) -> dict:
        """
        Smart rubric loading:
        - Default rubric (None) → Load pre-formatted JSON, skip formatter
        - Uploaded rubric → Use formatter agent with retry
        """

        if rubric_input is None:
            # Default rubric - already formatted, no agent needed
            print("  → Using default rubric (pre-formatted JSON)")
            rubric_data = load_default_rubric()

            # Validate it's already formatted
            if isinstance(rubric_data, dict) and "criteria" in rubric_data:
                print("  ✓ Default rubric loaded (formatter agent NOT called)")
                return rubric_data
            else:
                # Shouldn't happen, but handle just in case
                print("  ⚠ Default rubric not in expected format, using formatter...")
                return self._format_rubric_with_retry(str(rubric_data), max_retries)

        else:
            # Custom uploaded rubric - needs formatting
            print("  → Custom rubric uploaded, calling formatter agent...")
            raw_rubric = parse_uploaded_rubric(rubric_input)

            # Check if uploaded JSON is already formatted
            if isinstance(raw_rubric, dict) and "criteria" in raw_rubric:
                print("  ✓ Uploaded rubric is already formatted (JSON)")
                return raw_rubric
            else:
                # Needs formatting (Excel or unstructured)
                print("  → Formatting with RubricFormatter agent...")
                return self._format_rubric_with_retry(str(raw_rubric), max_retries)

    def _format_rubric_with_retry(self, raw_rubric: str, max_retries: int = 3) -> dict:
        """
        Format rubric with retry logic for invalid JSON responses.
        Only called for custom/uploaded rubrics that need formatting.
        """
        formatter = self._get_formatter_agent()  # Lazy load

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Formatting attempt {attempt}/{max_retries}")

                formatted = formatter.format_rubric(raw_rubric)

                # Validate structure
                if not isinstance(formatted, dict) or "criteria" not in formatted:
                    raise ValueError("Invalid rubric structure: missing 'criteria'")

                if not isinstance(formatted["criteria"], list):
                    raise ValueError("Invalid rubric: 'criteria' must be a list")

                print(f"  ✓ Format valid on attempt {attempt}")
                return formatted

            except (json.JSONDecodeError, ValueError) as e:
                print(f"  ⚠ Format error on attempt {attempt}: {e}")

                if attempt == max_retries:
                    print(f"  ❌ Max retries reached for rubric formatting")
                    raise ValueError(
                        f"Failed to format rubric after {max_retries} attempts: {e}"
                    )

                # Exponential backoff
                wait_time = 2**attempt
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                raise

    def _score_with_retry(
        self, transcript: str, rubric: dict, duration: int, max_retries: int = 3
    ) -> dict:
        """Score transcript with retry logic for invalid responses."""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Scoring attempt {attempt}/{max_retries}")

                result = self.scorer_agent.score(transcript, rubric, duration)

                # Validate structure
                required_fields = ["overall_score", "word_count", "per_criterion"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")

                # Validate score range
                if not (0 <= result["overall_score"] <= 100):
                    raise ValueError(f"Invalid score: {result['overall_score']}")

                # Validate per_criterion structure
                if not isinstance(result["per_criterion"], list):
                    raise ValueError("per_criterion must be a list")

                for criterion in result["per_criterion"]:
                    required_criterion_fields = [
                        "criterion",
                        "metric",
                        "score",
                        "max_score",
                        "feedback",
                        "details",
                    ]
                    for field in required_criterion_fields:
                        if field not in criterion:
                            raise ValueError(f"Missing field in criterion: {field}")

                print(f"  ✓ Scoring valid on attempt {attempt}")
                return result

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  ⚠ Scoring error on attempt {attempt}: {e}")

                if attempt == max_retries:
                    print(f"  ❌ Max retries reached for scoring")
                    raise ValueError(
                        f"Failed to score after {max_retries} attempts: {e}"
                    )

                # Exponential backoff
                wait_time = 2**attempt
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                raise

    def _validate_results(self, results: dict):
        """Final validation of complete results."""
        if not results:
            raise ValueError("Results are empty")

        if results["overall_score"] < 0 or results["overall_score"] > 100:
            raise ValueError(f"Invalid overall score: {results['overall_score']}")

        if results["word_count"] < 0:
            raise ValueError(f"Invalid word count: {results['word_count']}")

        if not results["per_criterion"]:
            raise ValueError("No per-criterion results found")
