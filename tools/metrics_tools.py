import re
from typing import Dict, List

import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize tools once (global)
grammar_tool = language_tool_python.LanguageTool("en-US")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Filler words list
FILLER_WORDS = [
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


def count_words(text: str) -> dict:
    """Count total words in text"""
    words = text.split()
    return {"word_count": len(words)}


def calculate_wpm(text: str, duration_seconds: int) -> dict:
    """
    Calculate Words Per Minute (WPM) and categorize.

    Args:
        text: Transcript text
        duration_seconds: Speech duration in seconds

    Returns:
        dict with wpm, category, and score
    """
    words = text.split()
    word_count = len(words)

    if duration_seconds <= 0:
        return {
            "error": "Duration must be positive",
            "wpm": None,
            "category": None,
            "score": None,
        }

    wpm = (word_count / duration_seconds) * 60

    # Categorize
    if wpm >= 161:
        category, score = "Too Fast", 2
    elif wpm >= 141:
        category, score = "Fast", 6
    elif wpm >= 111:
        category, score = "Ideal", 10
    elif wpm >= 81:
        category, score = "Slow", 6
    else:
        category, score = "Too Slow", 2

    return {"wpm": round(wpm, 2), "category": category, "score": score}


def check_grammar(text: str) -> dict:
    """
    Check grammar errors using LanguageTool.

    Returns grammar score based on: 1 - min(errors_per_100_words / 10, 1)
    """
    try:
        matches = grammar_tool.check(text)
        error_count = len(matches)

        words = text.split()
        word_count = len(words)

        errors_per_100 = (error_count / word_count * 100) if word_count > 0 else 0

        # Formula from rubric
        grammar_score = max(0, 1 - min(errors_per_100 / 10, 1))

        # Convert to 0-10 scale for rubric weight of 10
        final_score = grammar_score * 10

        return {
            "error_count": error_count,
            "errors_per_100_words": round(errors_per_100, 2),
            "grammar_score": round(final_score, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_ttr(text: str) -> dict:
    """
    Calculate Type-Token Ratio (vocabulary richness).
    TTR = unique_words / total_words
    """
    words = [w.lower() for w in text.split()]

    if not words:
        return {"ttr": 0, "score": 0}

    unique_words = len(set(words))
    total_words = len(words)

    ttr = unique_words / total_words

    # Score based on TTR ranges
    if ttr >= 0.9:
        score = 10
    elif ttr >= 0.7:
        score = 8
    elif ttr >= 0.5:
        score = 6
    elif ttr >= 0.3:
        score = 4
    else:
        score = 2

    return {
        "ttr": round(ttr, 3),
        "unique_words": unique_words,
        "total_words": total_words,
        "score": score,
    }


def detect_filler_words(text: str) -> dict:
    """
    Detect and count filler words.
    Returns filler rate = (filler_count / total_words) * 100
    """
    text_lower = " " + text.lower() + " "
    words = text.split()
    word_count = len(words)

    filler_found = []
    filler_count = 0

    for filler in FILLER_WORDS:
        # Count occurrences
        pattern = f" {filler} | {filler}, | {filler}. | {filler}\n"
        count = text_lower.count(f" {filler} ")
        count += text_lower.count(f" {filler},")
        count += text_lower.count(f" {filler}.")

        if count > 0:
            filler_found.append(filler)
            filler_count += count

    filler_rate = (filler_count / word_count * 100) if word_count > 0 else 0

    # Score based on filler rate
    if filler_rate <= 3:
        score = 15
    elif filler_rate <= 6:
        score = 12
    elif filler_rate <= 9:
        score = 9
    elif filler_rate <= 12:
        score = 6
    else:
        score = 3

    return {
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 2),
        "filler_words_found": filler_found,
        "score": score,
    }


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment using VADER.
    Returns positive probability and score.
    """
    try:
        scores = sentiment_analyzer.polarity_scores(text)
        positive_prob = scores["pos"]

        # Score based on positive probability
        if positive_prob >= 0.9:
            score = 15
        elif positive_prob >= 0.7:
            score = 12
        elif positive_prob >= 0.5:
            score = 9
        elif positive_prob >= 0.3:
            score = 6
        else:
            score = 3

        return {
            "positive": scores["pos"],
            "neutral": scores["neu"],
            "negative": scores["neg"],
            "compound": scores["compound"],
            "score": score,
        }
    except Exception as e:
        return {"error": str(e)}


def find_keywords(text: str, keywords: list) -> dict:
    """
    Find which keywords are present in text (case-insensitive).

    Args:
        text: Transcript text
        keywords: List of keywords to search for

    Returns:
        dict with found keywords
    """
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]

    return {
        "keywords_searched": keywords,
        "keywords_found": found,
        "found_count": len(found),
    }
