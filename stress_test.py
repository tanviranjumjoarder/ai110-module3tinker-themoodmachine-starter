"""
Stress tests ("breakers") for the Mood Machine.

These sentences are NOT in SAMPLE_POSTS on purpose: they are held out so we
can see how both models do on language they were not built around.

Run:
    python stress_test.py          # rule-based model only
    python stress_test.py --ml     # also train the ML model on SAMPLE_POSTS
                                   # and test it on the same breakers
"""

import sys
from typing import List, Tuple

from mood_analyzer import MoodAnalyzer

# (sentence, label a person would give it, what it is testing)
BREAKERS: List[Tuple[str, str, str]] = [
    ("I love getting stuck in traffic", "negative", "sarcasm"),
    ("Oh great, another Monday", "negative", "sarcasm"),
    ("Wow, thanks for nothing", "negative", "sarcasm"),
    ("that concert was sick", "positive", "slang"),
    ("this pizza is fire 🔥", "positive", "slang + emoji"),
    ("this song slaps", "positive", "slang"),
    ("I'm fine 🙂", "neutral", "emoji, possibly passive-aggressive"),
    ("best day ever 😂😂", "positive", "emoji"),
    ("missed the bus again 😭", "negative", "emoji only"),
    ("I'm exhausted but proud of myself", "mixed", "mixed emotions"),
    ("I can't stop smiling", "positive", "double negative"),
    ("not gonna lie, that was amazing", "positive", "negation phrase that isn't negation"),
    ("I feel sick today", "negative", "literal 'sick'"),
    ("the kitchen is on fire", "negative", "literal 'fire'"),
]

# Same meaning, different language style. If the model gets the standard
# version right but the informal one wrong, errors cluster around who wrote
# the text (overt bias), not what it says.
STYLE_PAIRS: List[Tuple[str, str, str]] = [
    # (standard wording, informal / regional / non-standard wording, label)
    ("This party is really fun", "this party is bussin fr", "positive"),
    ("That test was terrible", "that test was trash ngl", "negative"),
    ("I am very happy", "im sooo happyyy", "positive"),
    ("That was really good", "that was wicked good", "positive"),
    ("I am so tired", "I'm absolutely knackered", "negative"),
    ("The food was amazing", "the food was gr8", "positive"),
    ("I am sad today", "feeling some type of way today", "negative"),
]


def run_rule_based(analyzer: MoodAnalyzer) -> None:
    print("=== Breakers (rule based) ===")
    misses = 0
    for text, expected, kind in BREAKERS:
        got = analyzer.predict_label(text)
        ok = got == expected
        misses += not ok
        print(f'{"OK  " if ok else "MISS"} [{kind}] "{text}" -> {got} (expected {expected})')
        print(f"       {analyzer.explain(text)}")
    print(f"Breakers correct: {len(BREAKERS) - misses}/{len(BREAKERS)}\n")

    print("=== Language-style pairs (rule based) ===")
    std_ok = inf_ok = 0
    for standard, informal, expected in STYLE_PAIRS:
        a = analyzer.predict_label(standard)
        b = analyzer.predict_label(informal)
        std_ok += a == expected
        inf_ok += b == expected
        flag = "" if a == b else "   <-- different"
        print(f'"{standard}" -> {a} | "{informal}" -> {b} (expected {expected}){flag}')
    n = len(STYLE_PAIRS)
    print(f"Standard wording correct: {std_ok}/{n}; informal wording correct: {inf_ok}/{n}\n")


def run_ml() -> None:
    from dataset import SAMPLE_POSTS, TRUE_LABELS
    from ml_experiments import train_ml_model

    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    def predict(texts: List[str]) -> List[str]:
        return list(model.predict(vectorizer.transform(texts)))

    print("=== Breakers (ML, trained on SAMPLE_POSTS) ===")
    preds = predict([t for t, _, _ in BREAKERS])
    correct = 0
    for (text, expected, kind), got in zip(BREAKERS, preds):
        ok = got == expected
        correct += ok
        print(f'{"OK  " if ok else "MISS"} [{kind}] "{text}" -> {got} (expected {expected})')
    print(f"Breakers correct: {correct}/{len(BREAKERS)}\n")

    print("=== Language-style pairs (ML) ===")
    std = predict([s for s, _, _ in STYLE_PAIRS])
    inf = predict([i for _, i, _ in STYLE_PAIRS])
    std_ok = inf_ok = 0
    for (standard, informal, expected), a, b in zip(STYLE_PAIRS, std, inf):
        std_ok += a == expected
        inf_ok += b == expected
        flag = "" if a == b else "   <-- different"
        print(f'"{standard}" -> {a} | "{informal}" -> {b} (expected {expected}){flag}')
    n = len(STYLE_PAIRS)
    print(f"Standard wording correct: {std_ok}/{n}; informal wording correct: {inf_ok}/{n}\n")


if __name__ == "__main__":
    run_rule_based(MoodAnalyzer())
    if "--ml" in sys.argv:
        run_ml()
