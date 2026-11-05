# Model Card: Mood Machine

This model card covers **two** versions of the Mood Machine mood classifier:

1. A **rule-based model** in `mood_analyzer.py`
2. A **machine learning model** in `ml_experiments.py` (scikit-learn)

All numbers below come from running `python main.py`, `python ml_experiments.py` and `python stress_test.py --ml` on the final version of this repo.

## 1. Model Overview

**Model type:**
I compared both models. The rule-based model is the main one; the ML model is used as a comparison.

**Intended purpose:**
Label a short, informal post (one or two sentences, like a social media update or text message) as `positive`, `negative`, `neutral` or `mixed`. It is a classroom tool for seeing how data and rules shape predictions. It is not meant for real decisions about people.

**How it works (brief):**

- *Rule-based:* clean the text into tokens, add +1 for each positive word or emoji and −1 for each negative one, flip a word's sign if a negation word came just before it, then turn the total into a label (rules in section 3).
- *ML:* turn each post into word counts (bag of words), then fit a logistic regression on the 20 labeled posts so it learns which words go with which label.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **20** posts: the 6 starter posts, 10 I added in Part 1 and 4 more I added in Part 4 to see how each model reacted to new data. Label counts: 7 positive, 7 negative, 3 neutral, 3 mixed.

There is also a separate held-out set in `stress_test.py`: 14 "breaker" sentences and 7 pairs of the same idea written in standard versus informal wording. Neither model was built or trained on these.

**Labeling process:**
I labeled each post by what I think the writer meant, not by the words it uses. Posts that were hard to label, where someone else could reasonably pick differently:

- `"Just got the internship offer!!! 😭🎉"` → **positive**. Here 😭 means happy tears, but the same emoji means sadness in other posts.
- `"I absolutely love getting stuck in traffic for an hour"` → **negative** (sarcasm), even though it contains "love".
- `"That movie was mid tbh"` → **negative**. "Mid" is mild, so some people would call this neutral.
- `"I hate how much I love this show 😂"` → **positive** (affectionate complaining). It could also be called mixed.
- `"This is fine"` → **neutral**. Depending on tone this could be resigned or sarcastic.

**Important characteristics of the dataset:**

- Slang: lowkey, no cap, mid, tbh, fire
- Emojis and emoticons: 😭 🎉 😩 🔥 😂 :)
- Sarcasm: the traffic post
- Mixed feelings: "tired but kind of hopeful", "stressed but kind of proud", "exam went terrible but at least the weekend is here"
- Negation: "not happy", "not bad at all", "can't stop smiling"

**Possible issues with the dataset:**

- It is tiny (20 posts), so one post can change the ML model a lot.
- Only 3 neutral and 3 mixed posts, so those labels are underrepresented.
- Every post is short, in English, and written in a US student / internet register. There is no regional English, no non-native English, no code-switching and no long text.
- All labels come from one person, so there is no check on how much other people would agree.

## 3. How the Rule-Based Model Works

**Scoring rules:**

1. **Preprocess** (`preprocess`): lowercase; keep emoticons (`:)`, `:(`) and emoji as their own tokens; drop other punctuation; remove apostrophes (`can't` → `cant`); shrink stretched words (`soooo happyyy` → `so happy`, keeping whichever version is a known word).
2. **Word and emoji matches** (`analyze`): +1 for each word in `POSITIVE_WORDS`, −1 for each word in `NEGATIVE_WORDS`. Emoji in `EMOJI_SCORES` add their own ±1.
3. **Negation (my Part 2 enhancement):** if a negation word (`not`, `no`, `never`, `dont`, `cant`, `isnt`, ...) appeared within the previous 3 tokens, the next sentiment word counts the opposite way. So `not happy` = −1 and `not bad` = +1. Each negation flips one word only.
4. **Emoji and slang signals (my Part 3 fix):** added `EMOJI_SCORES` and the slang words `fire, slaps, bussin, goated, lit` (positive) and `mid, trash` (negative). Negation does not flip emoji, because in "fire no cap 🔥" the "no" is slang and doesn't negate anything. I left out 🙂 and 💀 on purpose: 🙂 is often passive-aggressive and 💀 usually means "dying of laughter".
5. **"can't stop" fix (Part 4):** a negated `stop` cancels the negation, so "can't stop smiling" stays positive.
6. **Thresholds** (`predict_label`): if both positive and negative signals fired and |score| ≤ 1 → `mixed`; otherwise score > 0 → `positive`, < 0 → `negative`, 0 → `neutral`.

`explain()` uses the same `analyze()` code as `predict_label()`, so the explanation printed in `main.py` always matches the actual prediction. Example:
`"Not bad at all, honestly better than I expected"` → `score=2 positive=['not bad', 'better'] negative=[]`.

**Strengths:**

- Every prediction can be traced back to specific words, so it is easy to debug.
- Plain sentences and plain negation work: "I am not happy about this", "Not bad at all...", "I'm exhausted but proud of myself" (mixed).
- It generalizes the same way to new text: it got 8/14 held-out breakers, while the ML model got 7/14.

**Weaknesses:**

- Words or emoji that aren't in a list are invisible: "Missing home a lot lately" scores 0 and is labeled neutral.
- It can't read sarcasm (section 5).
- One word has one fixed meaning, so slang and literal uses clash ("fire").
- `but` is treated like `and`, so the order and weight of a contrast are lost.

## 4. How the ML Model Works

**Features used:** bag of words from `CountVectorizer` with its default settings. That tokenizer keeps only words of 2+ letters, so **every emoji and emoticon is thrown away** before training. The vocabulary has 75 words and no emoji.

**Training data:** `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**

- Accuracy on the training posts is **1.00** every time, with 16 posts and with 20. That only shows it memorized the posts; it says nothing about new text.
- On the 14 held-out breakers it went from **5/14 → 7/14** after I added 4 posts. Its answers changed a lot. "this song slaps" went from positive to negative, and all three sarcasm breakers became "correct" only because the model now leans negative on almost anything unfamiliar.
- Label sensitivity test: I flipped the label of the traffic sarcasm post to `positive` and retrained. "I love getting stuck in traffic" flipped from negative to positive. One label out of 16 decided that prediction.

**Strengths and weaknesses:**

- Strength: it picked up sarcasm-like and slang patterns that I never wrote rules for, as long as similar words were in the training posts.
- Weakness: it learns spurious cues. "I am very happy" → **negative**, because "am" appears in "I am not happy about this". "best day ever 😂😂" → negative, because the only word it recognizes is "day", from "Today was a terrible day".

## 5. Evaluation

**How I evaluated:**

- `python main.py`: rule-based accuracy on the 20 labeled posts.
- `python ml_experiments.py`: ML accuracy on the same 20 posts. This is training accuracy, so it is optimistic.
- `python stress_test.py --ml`: both models on 14 held-out breakers and 7 standard-vs-informal wording pairs.

**Rule-based accuracy history:**

| Change | Posts | Accuracy | Breakers |
|---|---|---|---|
| Basic scoring + negation (Part 2) | 16 | 0.62 | 3/14 |
| + emoji and slang signals (Part 3 fix) | 16 | 0.75 | 7/14 |
| + "can't stop" negation fix | 16 | 0.81 | 8/14 |
| + 4 more posts (Part 4) | 20 | **0.75** | 8/14 |

The drop at the end comes from adding harder posts, not from making the model worse.

**Final comparison:**

| | Rule-based | ML |
|---|---|---|
| Labeled posts (20) | 0.75 | 1.00 (training data) |
| Held-out breakers (14) | 8/14 | 7/14 |
| Standard wording (7) | 7/7 | 4/7 |
| Informal wording (7) | 4/7 | 4/7 |

**Examples of correct predictions (rule-based):**

- `"I am not happy about this"` → negative. Negation flips `happy` to −1.
- `"Feeling tired but kind of hopeful"` → mixed. `tired` −1 and `hopeful` +1 cancel, and both sides fired.
- `"missed the bus again 😭"` → negative. Before the Part 3 fix this was neutral, because nothing matched; now 😭 = −1.

**Examples of incorrect predictions:**

- `"I absolutely love getting stuck in traffic for an hour"` (true: negative) → rule-based says **mixed**: `love` +1 and `stuck` −1 cancel. The ML model gets it right only because this exact post is in its training data.
- `"Just got the internship offer!!! 😭🎉"` (true: positive) → rule-based says **mixed**: 😭 −1 and 🎉 +1. The emoji fix caused this one, since 😭 usually means sad.
- `"Exam went terrible but at least the weekend is here"` (true: mixed) → **negative**. "weekend is here" is positive in meaning, but none of its words are in the word list.
- `"I hate how much I love this show 😂"` (true: positive) → rule-based says **mixed**: the literal words point both ways.

**Fix or document?**

- *"can't stop smiling" → negative.* My own negation rule caused this, so it was a correctable defect. **Fixed** it with the "stop" rule and checked that "I am not happy" is still negative.
- *Sarcasm.* Opposite meaning with the same words is a basic limit of counting words. **Documented**, not fixed. A rule like "love + stuck = sarcasm" would only fit this one sentence.
- *😭 happy tears vs sad tears.* The same token means opposite things. **Documented.**
- *"Missing home a lot lately".* I could add "missing", but then "missing the point" or "nothing missing" would be wrong. **Documented** as a vocabulary gap.

## 6. Limitations

- **Sarcasm:** "Oh great, another Monday" → positive (`great` +1). "I love getting stuck in traffic" → mixed.
- **Literal vs slang meanings:** after the slang fix, "the kitchen is on fire" → **positive** (before, it was neutral). I chose not to add "sick" for the same reason: "that concert was sick" stays neutral, so that "I feel sick today" doesn't turn positive.
- **Fixed-phrase negation:** "not gonna lie, that was amazing" is correct only because `amazing` comes 5 tokens after `not`, outside the 3-token window. Change the wording slightly and "not" would flip it.
- **Vocabulary coverage:** anything outside the lists is neutral, e.g. "the food was gr8", "I'm absolutely knackered", "feeling some type of way today".
- **Tiny data:** 20 labeled posts from one person. The ML model's 1.00 is training accuracy; on held-out text it gets 7/14.
- **Short text only:** never tested on paragraphs, where a total score would mostly reflect length.

## 7. Ethical Considerations

**Overt bias (who the model works for):**
In the style-pair test, the rule-based model got **7/7 standard-wording posts but only 4/7 informal ones**, even after the slang fix (it was 2/7 before). Errors cluster around **writers who use non-standard spelling, regional expressions or newer slang**, such as British English ("knackered"), texting shorthand ("gr8"), and phrases common in African American English and online communities ("feeling some type of way"). Those posts aren't less emotional; they are just missing from my word lists. The slang I did add is US, Gen-Z, internet slang, which is my own register, so the model works best for people who write like me. The ML model has the same problem, plus it barely handled standard wording (4/7) because it learned from so few examples.

**Other risks:**

- Labeling a distress message as neutral or positive ("I'm fine 🙂") could mean someone who needs help gets missed if a tool like this were used for moderation or wellbeing checks.
- Running mood detection on personal messages raises privacy concerns, even with a model this simple.

**Covert compliance (how the AI assistant behaved):**
I used an AI coding assistant to plan and write the analyzer, suggest breaker sentences and draft this card.

- *Where it raised concerns:* it pointed out that 🙂 and 💀 are ambiguous, and that adding "sick" or "fire" would break literal uses. It kept "sick" out and added "fire" knowingly, then showed the kitchen-fire regression.
- *Where it didn't push back:* it accepted my labels as given, for example "mid" = negative and 😭 = happy tears in one post but sad in the emoji list. It also didn't question that every slang word it added was American internet slang. The regional and dialect gap only came up because the lab told me to test language styles, not because the assistant raised it.
- *What it should have flagged earlier:* the ML vectorizer silently drops every emoji. That makes the "ML vs rules on emoji" comparison unfair, and it only came out when I inspected the vocabulary.

## 8. Ideas for Improvement

- Collect more labeled posts from different writers, have more than one person label each one, and measure agreement.
- Evaluate on a real held-out test set, not training accuracy.
- Keep emoji in the ML model: `CountVectorizer(token_pattern=r"[^\s]+")` or a custom tokenizer.
- Weight the clause after "but" more heavily ("terrible **but** at least the weekend is here").
- End negation scope at punctuation instead of using a fixed 3-token window.
- Add regional and dialect vocabulary, tested by people who actually use it.
- Try TF-IDF or a small pretrained sentiment model, and compare it on the same breakers.
