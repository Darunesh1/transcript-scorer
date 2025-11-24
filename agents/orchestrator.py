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
    Orchestrator Agent using ADK patterns from Kaggle Day 5.
    Coordinates between RubricFormatter and Scorer agents with retry logic.
    """

    def __init__(self):
        print("🚀 Initializing Orchestrator Agent with ADK patterns...")

        # Initialize client with retry options (from ADK best practices)
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=1,
                    max_delay=10,
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

        # Initialize sub-agents
        self.formatter_agent = RubricFormatterAgent(self.client)
        self.scorer_agent = ScoringAgent(self.client)

        print("✓ Orchestrator initialized with retry logic")
        print("✓ Sub-agents: RubricFormatter, Scorer")

    def process(
        self, transcript_input, rubric_input=None, duration=None, max_retries=3
    ):
        """
        Main orchestration with retry logic for format errors.

        Workflow (ADK Multi-Agent Pattern):
        1. Extract transcript text
        2. Load rubric data
        3. Format rubric (Agent 1) with retry
        4. Score transcript (Agent 2) with retry
        5. Validate and return results
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

            # Step 2: Load rubric
            print("\n📋 Step 2: Loading rubric...")
            raw_rubric = self._load_rubric(rubric_input)
            print(f"✓ Rubric loaded: {len(raw_rubric)} chars")

            # Step 3: Format rubric with retry (Agent 1)
            print("\n🔧 Step 3: Formatting rubric (Agent 1)...")
            formatted_rubric = self._format_rubric_with_retry(raw_rubric, max_retries)
            print(
                f"✓ Rubric formatted: {len(formatted_rubric.get('criteria', []))} criteria"
            )

            # Step 4: Score with retry (Agent 2)
            print("\n🎯 Step 4: Scoring transcript (Agent 2)...")
            results = self._score_with_retry(
                transcript, formatted_rubric, duration, max_retries
            )
            print(f"✓ Scoring complete: {results['overall_score']:.1f}/100")

            # Step 5: Validate final results
            print("\n✅ Step 5: Validating results...")
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

    def _load_rubric(self, rubric_input):
        """Load rubric from default or uploaded source."""
        if rubric_input is None:
            return load_default_rubric()
        else:
            return parse_uploaded_rubric(rubric_input)

    def _format_rubric_with_retry(self, raw_rubric: str, max_retries: int = 3) -> dict:
        """
        Format rubric with retry logic for invalid JSON responses.
        ADK Pattern: Retry with exponential backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Attempt {attempt}/{max_retries}")

                formatted = self.formatter_agent.format_rubric(raw_rubric)

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
        """
        Score transcript with retry logic for invalid responses.
        ADK Pattern: Validate output and retry on format errors.
        """
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  → Attempt {attempt}/{max_retries}")

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
