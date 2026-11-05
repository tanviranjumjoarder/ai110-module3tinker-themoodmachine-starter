# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── stress_test.py     # Held-out breaker sentences + standard vs informal wording pairs
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.

---

## My Results

Run everything (after `pip install -r requirements.txt`):

```bash
python main.py              # rule-based evaluation with per-post explanations
python ml_experiments.py    # ML model (training accuracy)
python stress_test.py --ml  # both models on held-out breakers and style pairs
```

| | Rule-based | ML |
|---|---|---|
| 20 labeled posts | 0.75 | 1.00 (training data) |
| 14 held-out breakers | 8/14 | 7/14 |
| Standard vs informal wording | 7/7 vs 4/7 | 4/7 vs 4/7 |

What I built: text cleanup (emoji tokens, stretched words, apostrophes), negation
handling, emoji and slang signals, a `mixed` label, and explanations that show
which words drove each score. See [`model_card.md`](model_card.md) for the full
evaluation, the failures I fixed versus documented, and the bias analysis.
