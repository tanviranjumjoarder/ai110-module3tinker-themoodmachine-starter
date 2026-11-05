# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

Pipeline:
  - preprocess(): clean the text and split it into tokens
  - analyze():    walk the tokens, match words, apply negation, total a score
  - score_text(): the numeric score from analyze()
  - predict_label(): map the score (and which signals fired) to a label
  - explain():    human readable reasons for a prediction
"""

import re
from typing import Dict, List, Optional

from dataset import EMOJI_SCORES, NEGATIVE_WORDS, POSITIVE_WORDS

# Words that flip the meaning of the next sentiment word ("not happy").
# Apostrophes are removed in preprocess, so "don't" arrives as "dont".
NEGATION_WORDS = {
    "not", "no", "never", "nothing", "hardly",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent",
    "cant", "cannot", "couldnt", "wont", "wouldnt", "shouldnt", "aint",
}

# How many tokens after a negation word we still look for a sentiment
# word to flip. "not very happy" needs 2; "not bad at all" needs 1.
NEGATION_WINDOW = 3

# One regex that pulls out, in order:
#   1. ASCII emoticons like :) :-( ;) :D :P (after lowercasing: :d :p)
#   2. single emoji characters (most emoji live in these Unicode blocks)
#   3. words, keeping apostrophes so "can't" stays one token for now
TOKEN_PATTERN = re.compile(
    r"[:;=]-?[()dp/|]"
    r"|[\U0001F300-\U0001FAFF☀-➿]"
    r"|[a-z0-9']+"
)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)
        self.emoji_scores = dict(EMOJI_SCORES)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

          - lowercase
          - keep emoticons (":)") and emoji ("🔥") as their own tokens
          - drop other punctuation
          - remove apostrophes: "can't" -> "cant", "I'm" -> "im"
          - shrink stretched words: "soooo happyyy" -> "so happy"
        """
        raw_tokens = TOKEN_PATTERN.findall(text.strip().lower())

        tokens: List[str] = []
        for tok in raw_tokens:
            if tok[0].isalnum() or tok[0] == "'":
                tok = tok.replace("'", "")
                if not tok:
                    continue
                tok = self._normalize_repeats(tok)
            tokens.append(tok)
        return tokens

    def _normalize_repeats(self, word: str) -> str:
        """Collapse runs of 3+ identical letters, preferring a known word."""
        if not re.search(r"(.)\1\1", word):
            return word
        single = re.sub(r"(.)\1{2,}", r"\1", word)   # "happyyy" -> "happy"
        double = re.sub(r"(.)\1{2,}", r"\1\1", word)  # "goooood" -> "good"
        known = self.positive_words | self.negative_words
        if single in known:
            return single
        if double in known:
            return double
        return single

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """
        Do the actual scoring and keep the evidence.

        Rules:
          - positive word -> +1, negative word -> -1
          - ENHANCEMENT (negation): if a negation word ("not", "never",
            "dont", ...) appeared within the previous NEGATION_WINDOW tokens,
            the next sentiment word counts the opposite way, so
            "not happy" = -1 and "not bad" = +1.
          - FIX (Part 3, emoji signals): emoji/emoticons in EMOJI_SCORES add
            their score. Negation does NOT flip them: in "fire no cap 🔥"
            the "no" is slang, and an emoji is a separate signal anyway.
        """
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0
        tokens_since_negation: Optional[int] = None

        for token in tokens:
            if token in self.emoji_scores:
                value = self.emoji_scores[token]
                score += value
                (positive_hits if value > 0 else negative_hits).append(token)
                continue

            if token in NEGATION_WORDS:
                tokens_since_negation = 0
                continue

            if token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                value = 0

            negated = (
                tokens_since_negation is not None
                and tokens_since_negation < NEGATION_WINDOW
            )
            if value != 0:
                label = token
                if negated:
                    value = -value
                    label = f"not {token}"
                    tokens_since_negation = None  # one negation flips one word
                score += value
                (positive_hits if value > 0 else negative_hits).append(label)
            elif tokens_since_negation is not None:
                tokens_since_negation += 1

        return {
            "tokens": tokens,
            "score": score,
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
        }

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive signals add points, negative signals subtract points,
        negation flips the next signal. See analyze() for the details.
        """
        return self.analyze(text)["score"]

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the score for a piece of text into a mood label.

          - both positive and negative signals AND |score| <= 1 -> "mixed"
            ("tired but kind of hopeful": the two sides roughly cancel)
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - otherwise  -> "neutral" (nothing in the text matched)
        """
        result = self.analyze(text)
        score = result["score"]
        if result["positive_hits"] and result["negative_hits"] and abs(score) <= 1:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label,
        using exactly the same logic as predict_label.

        Example: 'score=1 positive=['not bad', 'better'] negative=[]'
        """
        result = self.analyze(text)
        return (
            f"score={result['score']} "
            f"positive={result['positive_hits']} "
            f"negative={result['negative_hits']} "
            f"tokens={result['tokens']}"
        )
