# ScienceQA evaluation protocol

Fixed before feature extraction, model selection, or test evaluation on 27 October 2026. The configuration and source hashes are recorded by the implementation before execution. This is the implementation of Qingyan Han's own Project 2.1 proposal, not Group 2's PetFinder project.

## Population and inputs

Use the cached `derek-thomas/ScienceQA` snapshot recorded in config.json, retaining the official training, validation and test splits. Include rows with an image and two to four choices. Every exclusion is counted; unreadable included images cause a failure, not silent removal. Stable IDs are snapshot split plus original row index. Original IDs are not available in this conversion.

Only question, hint, choices and image are model inputs. Answer is the label. Lecture, solution, subject, grade, topic and category are not model inputs. Subject and grade are diagnostic metadata. Do not redistribute the source dataset or model weights.

Audit exact image bytes and normalised input text for repeats within and across splits. Preserve official denominators; report performance on unseen-image and unseen-text test subsets as diagnostics, not new independent test sets. Repeated illustrations alone do not prove identical questions or answer leakage.

## Features and models

Frozen ImageNet ResNet18 produces 512-dimensional image features. Preserve the complete diagram with aspect-preserving resize and white padding to 224x224, followed by ImageNet normalisation. This differs from the standard centre crop to avoid removing diagram labels. No image augmentation or encoder fine-tuning.

Frozen cached multilingual MiniLM-L12-v2 produces 384-dimensional sentence embeddings. Encode question plus hint as one sequence and each ordered answer choice separately (up to128 tokens per sequence). Concatenate five embeddings: question/hint, A, B, C, D; absent choices use zero vectors. The 1920-dimensional representation preserves choice identity. Record truncation incidence. This is an explicit implementation detail of the proposal's sentence-embedding approach. A multilingual encoder is used because it is available locally; it does not make an English-only evaluation multilingual.

Baselines: training-majority answer index, expected uniform random accuracy (mean 1/K), and train-fitted TF-IDF plus logistic regression. Choose C from the fixed three-value grid using validation negative log-likelihood (NLL), not test performance.

Text-only, image-only and early fusion use the same 128-unit ReLU/dropout MLP head architecture, with four output logits. Early fusion concatenates standardised image and text features before the head. Each feature scaler is fitted only on training data. Late fusion combines the selected text and image head probabilities, using a text weight from the fixed grid selected by validation NLL. Endpoints are allowed: collapse to a unimodal model is a valid negative result.

Every method masks answer indices >=the row's actual choice count and renormalises over legal choices. No ground-truth-dependent mask. Train with unweighted masked cross-entropy, AdamW, fixed configuration and seeds42/43/44. Stop after6 epochs without validation-NLL improvement of0.0001, or40 epochs. Restore the best checkpoint. Same official splits for every seed; these are optimisation-seed repeats, not independent dataset resamples.

## Selection and evaluation

Primary seed42 is fixed in advance; seeds43/44 measure optimisation sensitivity. Save training and validation curves for every head, every candidate's validation result, and a selection lock before opening test features/labels for evaluation. No new hyperparameters or model variants after test evaluation. An execution error may be repaired only with an explicit receipt; never overwrite a failed run silently.

Report accuracy, macro-F1 over answer positions0–3, NLL and multiclass Brier score; position macro-F1 is a diagnostic rather than semantic answer-category fairness. Save per-sample probabilities and labels. Uniform random accuracy is an expectation; do not attach an arbitrary argmax F1 to it. Show confusion matrices, train/validation gaps, seed sensitivity, and subject/grade/choice-count groups with denominators.

Predefined ablation: permute image features within choice-count strata using seed2026 at test evaluation, keeping text and trained weights fixed. Report early/late accuracy changes; do not treat distribution-shift sensitivity as proof of reasoning or causation. Paired question-level bootstrap (2000 draws, seed2026) of fusion-minus-text correctness yields descriptive 95% intervals on primary seed42; repeated templates/images limit IID interpretation.

Measure encoder extraction cost separately from head fitting and cached-feature inference. UseCPU,4PyTorch threads. Benchmark cached head inference after warm-up over20 repeats, report median ms/sample; report serialised checkpoint/scaler bytes separately from shared encoder weights. No GPU/T4 claims. Failure illustrations use the first two primary early-fusion errors by stable test order, plus the first case corrected by fusion if one exists, rather than cherry-picking the most attractive examples.

## Authorship and submission

Maintain AI_ASSISTANCE.md describing actual AI contributions. Course-specific AI permissions remain unverified because the linked external course-information page is blank; this package cannot certify submission compliance. The user must review, understand and accurately disclose assistance under the applicable course rules. Personal presentation and live Q&A cannot be fulfilled by claiming AI-generated material is the student's own recorded speech. Do not submit toCanvas, contact staff, change groups, or publish source data.
