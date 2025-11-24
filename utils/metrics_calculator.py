import re
from collections import Counter
from typing import Dict, List

import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class MetricsCalculator:
    """
    Fast Python-based metrics calculation.
    Pre-compute all metrics before sending to AI for scoring.
    """

    def __init__(self):
        # Initialize tools once (reuse across requests)
        self.grammar_tool = language_tool_python.LanguageTool("en-US")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Filler words list
        self.filler_words = [
            "um",
            "uh",
            "like",
            "you know",
            "so",
            "actually",
            "basically",
            "right",
            "i mean",
            "well",
            "kinda",
            "sort of",
            "okay",
            "hmm",
            "ah",
        ]

        print("  ✓ MetricsCalculator initialized (LanguageTool, VADER)")

    def calculate_all_metrics(self, transcript: str, duration: int = None) -> Dict:
        """
        Calculate all metrics at once.
        Returns dict with pre-computed values for fast AI scoring.
        """
        print("  → Calculating metrics with Python libraries...")

        # Basic counts
        words = transcript.split()
        word_count = len(words)
        sentences = re.split(r"[.!?]+", transcript)
        sentence_count = len([s for s in sentences if s.strip()])

        # 1. Salutation detection
        salutation = self._detect_salutation(transcript)

        # 2. Keyword presence
        keywords_found = self._detect_keywords(transcript)

        # 3. Flow detection
        flow_followed = self._check_flow(transcript)

        # 4. Speech rate (WPM)
        wpm = None
        wpm_category = None
        if duration and duration > 0:
            wpm = (word_count / duration) * 60
            wpm_category = self._categorize_wpm(wpm)

        # 5. Grammar score (using LanguageTool)
        grammar_score, grammar_errors = self._calculate_grammar_score(
            transcript, word_count
        )

        # 6. Vocabulary richness (TTR)
        ttr = self._calculate_ttr(words)

        # 7. Filler word rate
        filler_count, filler_rate, filler_found = self._calculate_filler_rate(
            transcript, word_count
        )

        # 8. Sentiment (VADER)
        sentiment_scores = self._calculate_sentiment(transcript)

        metrics = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "duration_seconds": duration,
            # Salutation
            "salutation": {
                "detected": salutation["level"],
                "score": salutation["score"],
                "keywords": salutation["keywords"],
            },
            # Keywords
            "keywords": {
                "must_have_found": keywords_found["must_have"],
                "good_to_have_found": keywords_found["good_to_have"],
                "must_have_count": len(keywords_found["must_have"]),
                "good_to_have_count": len(keywords_found["good_to_have"]),
            },
            # Flow
            "flow": {"followed": flow_followed, "score": 5 if flow_followed else 0},
            # WPM
            "wpm": {
                "value": wpm,
                "category": wpm_category,
                "score": self._get_wpm_score(wpm) if wpm else None,
            },
            # Grammar
            "grammar": {
                "score": grammar_score,
                "errors_count": grammar_errors,
                "errors_per_100": (grammar_errors / word_count * 100)
                if word_count > 0
                else 0,
            },
            # TTR
            "ttr": {"value": ttr, "score": self._get_ttr_score(ttr)},
            # Filler words
            "filler_words": {
                "count": filler_count,
                "rate": filler_rate,
                "found": filler_found,
                "score": self._get_filler_score(filler_rate),
            },
            # Sentiment
            "sentiment": {
                "positive": sentiment_scores["pos"],
                "neutral": sentiment_scores["neu"],
                "negative": sentiment_scores["neg"],
                "compound": sentiment_scores["compound"],
                "score": self._get_sentiment_score(sentiment_scores["pos"]),
            },
        }

        print(
            f"  ✓ Metrics calculated: WPM={wpm:.1f if wpm else 'N/A'}, TTR={ttr:.2f}, Grammar={grammar_score:.2f}"
        )
        return metrics

    def _detect_salutation(self, text: str) -> Dict:
        """Detect salutation level"""
        text_lower = text.lower()

        excellent_keywords = [
            "excited to introduce",
            "feeling great",
            "pleasure to introduce",
        ]
        good_keywords = [
            "good morning",
            "good afternoon",
            "good evening",
            "good day",
            "hello everyone",
        ]
        normal_keywords = ["hi", "hello", "hey"]

        for kw in excellent_keywords:
            if kw in text_lower:
                return {"level": "Excellent", "score": 5, "keywords": [kw]}

        for kw in good_keywords:
            if kw in text_lower:
                return {"level": "Good", "score": 4, "keywords": [kw]}

        for kw in normal_keywords:
            if kw in text_lower:
                return {"level": "Normal", "score": 2, "keywords": [kw]}

        return {"level": "No Salutation", "score": 0, "keywords": []}

    def _detect_keywords(self, text: str) -> Dict:
        """Detect must-have and good-to-have keywords"""
        text_lower = text.lower()

        must_have = ["name", "age", "school", "class", "family", "hobbies", "interest"]
        good_to_have = [
            "about family",
            "from",
            "parents",
            "ambition",
            "goal",
            "dream",
            "fun fact",
            "unique",
            "strengths",
        ]

        must_found = [kw for kw in must_have if kw in text_lower]
        good_found = [kw for kw in good_to_have if kw in text_lower]

        return {"must_have": must_found, "good_to_have": good_found}

    def _check_flow(self, text: str) -> bool:
        """Check if introduction follows proper order"""
        text_lower = text.lower()

        # Simple heuristic: check if name comes early, closing comes late
        has_salutation_early = any(
            kw in text_lower[:100] for kw in ["hello", "hi", "good morning"]
        )
        has_name_early = "name" in text_lower[:200] or "myself" in text_lower[:200]
        has_closing = any(
            kw in text_lower[-100:] for kw in ["thank", "thanks", "bye", "goodbye"]
        )

        return has_salutation_early and has_name_early and has_closing

    def _categorize_wpm(self, wpm: float) -> str:
        """Categorize WPM"""
        if wpm >= 161:
            return "Too Fast"
        elif wpm >= 141:
            return "Fast"
        elif wpm >= 111:
            return "Ideal"
        elif wpm >= 81:
            return "Slow"
        else:
            return "Too Slow"

    def _get_wpm_score(self, wpm: float) -> float:
        """Get score based on WPM"""
        if 111 <= wpm <= 140:
            return 10
        elif (81 <= wpm <= 110) or (141 <= wpm <= 160):
            return 6
        else:
            return 2

    def _calculate_grammar_score(self, text: str, word_count: int):
        """Calculate grammar score using LanguageTool"""
        try:
            matches = self.grammar_tool.check(text)
            error_count = len(matches)
            errors_per_100 = (error_count / word_count * 100) if word_count > 0 else 0

            # Formula from rubric: 1 - min(errors_per_100 / 10, 1)
            grammar_score = max(0, 1 - min(errors_per_100 / 10, 1))

            return grammar_score, error_count
        except:
            return 0.8, 0  # Fallback

    def _calculate_ttr(self, words: List[str]) -> float:
        """Calculate Type-Token Ratio"""
        if not words:
            return 0.0

        unique_words = len(set(word.lower() for word in words))
        total_words = len(words)

        return unique_words / total_words if total_words > 0 else 0.0

    def _get_ttr_score(self, ttr: float) -> float:
        """Get score based on TTR"""
        if ttr >= 0.9:
            return 10
        elif ttr >= 0.7:
            return 8
        elif ttr >= 0.5:
            return 6
        elif ttr >= 0.3:
            return 4
        else:
            return 2

    def _calculate_filler_rate(self, text: str, word_count: int):
        """Calculate filler word rate"""
        text_lower = text.lower()

        filler_found = []
        filler_count = 0

        for filler in self.filler_words:
            count = (
                text_lower.count(f" {filler} ")
                + text_lower.count(f" {filler},")
                + text_lower.count(f" {filler}.")
            )
            if count > 0:
                filler_found.append(filler)
                filler_count += count

        filler_rate = (filler_count / word_count * 100) if word_count > 0 else 0

        return filler_count, filler_rate, filler_found

    def _get_filler_score(self, filler_rate: float) -> float:
        """Get score based on filler rate"""
        if filler_rate <= 3:
            return 15
        elif filler_rate <= 6:
            return 12
        elif filler_rate <= 9:
            return 9
        elif filler_rate <= 12:
            return 6
        else:
            return 3

    def _calculate_sentiment(self, text: str) -> Dict:
        """Calculate sentiment using VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores

    def _get_sentiment_score(self, positive_prob: float) -> float:
        """Get score based on positive sentiment"""
        if positive_prob >= 0.9:
            return 15
        elif positive_prob >= 0.7:
            return 12
        elif positive_prob >= 0.5:
            return 9
        elif positive_prob >= 0.3:
            return 6
        else:
            return 3
