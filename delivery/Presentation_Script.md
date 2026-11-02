# ScienceQA presentation script

Prepared with AI assistance. Read, understand and adapt before recording. Synthetic narration is only a rehearsal.

## Slide 1

This project asks whether adding images improves science question answering when the encoders are frozen. The system receives a question, a hint, answer choices, and a diagram or photograph. It predicts an answer position. The aim is a reproducible, low-compute comparison of early and late fusion, rather than a claim of general scientific reasoning.

## Slide 2

The experiment keeps the official ScienceQA splits and includes only questions with an image and two to four options. This gives 6,108 training, 2,053 validation, and 1,979 test questions. ResNet eighteen supplies image features, while MiniLM encodes the question and ordered choices. Early fusion joins the features before a small neural classifier. Late fusion combines two classifiers' probabilities. Validation loss selects checkpoints and a text weight of seventy percent. The test set is evaluated after all choices are locked.

## Slide 3

On the primary seed, text alone reaches 43.86 percent accuracy. Early fusion reaches 42.40 percent, and late fusion 43.36 percent. Neither improves accuracy over text, and the difference intervals include zero. However, late fusion improves two probability error scores, negative log likelihood and Brier score. Lower scores mean better probability predictions. The same broad accuracy ordering remains when averaging three random seeds.

## Slide 4

Overfitting is visible: training loss keeps falling while validation loss worsens. The selected checkpoint is epoch two. Shuffling images reduces early fusion accuracy by only 0.66 percentage points. This suggests limited useful visual contribution, although it is not a causal test of reasoning. Repeated images also matter: 555 test images occur in training. Accuracy is lower on unseen images, so the full result should not be presented as independent generalisation.

## Slide 5

Two errors make the limitations concrete. In the first question, a long engineering passage is cut from 253 to 128 tokens. The retained text loses the actual wind-tunnel conditions. In the second, the model mistakes the highlighted New Hampshire region for Rhode Island. These examples were selected as the first two errors in test order, rather than chosen for visual appeal. They motivate better context handling and more spatially precise features.

## Slide 6

The conclusion is that these frozen image features do not improve answer accuracy. Late fusion does improve probability scores. Feature extraction takes about eleven minutes on four CPU threads, while fitting each small head takes under three seconds. The data have subject imbalance and repeated templates, and the model should not grade students autonomously. Future work should validate longer text and OCR-aware representations on fresh validation data. Code, saved predictions, and AI assistance are documented in the report.
