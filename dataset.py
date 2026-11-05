"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # added: everyday positive words that show up in the dataset
    "proud",
    "hopeful",
    "better",
    "smiling",
    "glad",
    "enjoy",
    "wonderful",
    # added in Part 3 (slang fix): common praise slang
    "fire",
    "slaps",
    "bussin",
    "goated",
    "lit",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # added: everyday negative words that show up in the dataset
    "stuck",
    "died",
    "worse",
    "annoyed",
    "exhausted",
    "lonely",
    # added in Part 3 (slang fix): common put-down slang
    "mid",
    "trash",
]

# Emoji and emoticon signals (added in Part 3).
# Left out on purpose because their meaning depends on context:
#   "🙂" (can be passive-aggressive) and "💀" (usually means "I'm dying laughing").
EMOJI_SCORES = {
    ":)": 1,
    ":d": 1,
    ";)": 1,
    "😂": 1,
    "😍": 1,
    "🎉": 1,
    "🔥": 1,
    "❤": 1,
    ":(": -1,
    "😩": -1,
    "😭": -1,
    "😡": -1,
    "😞": -1,
}

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- Added posts: slang, emojis, sarcasm, mixed feelings, subtle tone ---
    "Lowkey stressed but kind of proud of myself",
    "Just got the internship offer!!! 😭🎉",
    "I absolutely love getting stuck in traffic for an hour",
    "ugh my laptop died right before the deadline 😩",
    "the new album is fire no cap 🔥",
    "Going to the grocery store after class",
    "Not bad at all, honestly better than I expected",
    "Exam went terrible but at least the weekend is here",
    "can't stop smiling today :)",
    "That movie was mid tbh",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "positive",  # "Just got the internship offer!!! 😭🎉"  (😭 = happy tears here)
    "negative",  # "I absolutely love getting stuck in traffic..."  (sarcasm)
    "negative",  # "ugh my laptop died right before the deadline 😩"
    "positive",  # "the new album is fire no cap 🔥"  (slang praise)
    "neutral",   # "Going to the grocery store after class"
    "positive",  # "Not bad at all, honestly better than I expected"
    "mixed",     # "Exam went terrible but at least the weekend is here"
    "positive",  # "can't stop smiling today :)"
    "negative",  # "That movie was mid tbh"  (mild; some would say neutral)
]

# Every post needs exactly one label. Fail loudly here instead of letting
# zip() silently drop examples later in main.py / ml_experiments.py.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS has {len(SAMPLE_POSTS)} posts but TRUE_LABELS has "
    f"{len(TRUE_LABELS)} labels"
)
