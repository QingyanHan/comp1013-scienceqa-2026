"""Build the coursework report from measured, locked result files."""
import html
import json
import re
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
import fitz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery'; OUT.mkdir(exist_ok=True)
R=ROOT/'results'; F=ROOT/'figures'
read=lambda n:json.loads((R/n).read_text(encoding='utf-8'))
summary=read('analysis_summary.json'); selection=read('selection_lock.json')['selections']
metrics=pd.read_csv(R/'test_metrics.csv'); primary=metrics[metrics.seed==42].set_index('method')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=10.1,leading=14.1,spaceAfter=7,textColor=colors.HexColor('#182b3b')))
styles.add(ParagraphStyle(name='CaptionReport',fontName='Helvetica',fontSize=8.4,leading=11,spaceAfter=8,textColor=colors.HexColor('#455766')))
styles.add(ParagraphStyle(name='SmallReport',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=6))
styles['Title'].fontName='Helvetica-Bold'; styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].textColor=colors.black
styles['Heading1'].fontSize=14;styles['Heading1'].leading=18;styles['Heading1'].textColor=colors.black
story=[]; body_text=[]; collecting=True

def p(text,style='BodyReport'):
    story.append(Paragraph(text,styles[style]))
    if collecting:body_text.append(re.sub('<[^>]+>',' ',text))

def h(text):p(text,'Heading1')
def table(rows,widths):
    if collecting:body_text.extend(' '.join(str(x) for x in row) for row in rows)
    cells=[[Paragraph(html.escape(str(c)),styles['SmallReport']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f1')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#5e7c80')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#a6b4bc')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,7)])

def figure(stem,width,caption):
    im=Image(str(F/f'{stem}.png')); im.drawHeight=im.imageHeight*width/im.imageWidth;im.drawWidth=width
    story.append(im);p(caption,'CaptionReport')

p('ScienceQA answer prediction<br/>Early and late fusion with frozen encoders','Title')
p('Qingyan Han | a1988836 | COMP1013 Project 2.2 | 2 November 2026','CaptionReport')
h('1. Problem and data quality')
p('This project tests whether adding frozen visual features improves multiple-choice science answer prediction over a text model. The practical setting is a low-compute educational prototype that receives a question, optional hint, answer choices and an image. Predicting an answer position is narrower than explaining scientific reasoning. ScienceQA [1] provides the task; this experiment evaluates only its image-bearing subset and makes no comparison with full-benchmark leaderboard scores.')
p('The cached Hugging Face conversion [2] retains the official splits. Rows without an image or outside two to four choices are excluded before fitting. Table 1 reconciles every exclusion. All retained images decoded successfully, and all labels were legal. Missing hints remain empty rather than being filled with answers. Lectures, solutions and diagnostic metadata are excluded from model inputs. Stable identifiers retain the original split and row index.')
table([['Split','Original','No image','Other K','Retained'],['Train','12,726','6,508','110','6,108'],['Validation','4,241','2,144','44','2,053'],['Test','4,241','2,224','38','1,979']],[100,95,100,90,110])
p('Table 1. Population accounting. K is the number of answer options. The test population was processed only after model selection was locked.','CaptionReport')
p('<b>Changes from proposal.</b> The task, official splits and two fusion families are retained. Proposal counts 6,218/2,097/2,017 preceded the choice-count filter; Table 1 corrects all three denominators. The cached MiniLM model, explicit 128-unit heads and validation-selected settings make the planned architecture concrete. Three seeds, probability scores, overlap and truncation audits extend the planned accuracy/F1 evaluation to expose uncertainty and shortcuts. The proposal\'s projected T4 timing was an extrapolation, not a GPU measurement; this report replaces it with measured CPU costs. Appendix B supplies the promised confusion matrices.')
p('Exact-byte image and normalised full-input-text hashes reveal repeated material: 576 validation images and 429 validation texts occur in training. The test counts are 555 and 375. Repeated illustrations do not necessarily imply identical questions, but they limit independence. Official denominators are retained, with unseen-image/text subset diagnostics in Section 4. Training labels A/B/C/D number 2,244/2,153/1,153/558, so position imbalance matters.')
h('2. Representations and fusion')
p('The frozen ImageNet ResNet-18 encoder [3,4] supplies 512 features. RGB images are resized with aspect-preserving white padding to 224 by 224 and ImageNet normalisation. Padding retains diagram boundaries that centre cropping could remove. No OCR, augmentation or encoder fine-tuning is used.')
p('The cached multilingual MiniLM sentence encoder [5,6] supplies a normalised 384-dimensional embedding for question plus hint and each ordered choice. Concatenating five slots gives 1,920 features, with zero slots for missing options. Each sequence is capped at 128 tokens. Question/hint truncation affects 529/6,108 training, 161/2,053 validation and 174/1,979 test rows; no choices are truncated. Local availability motivates this encoder, while the evaluation remains English-only.')
story.append(PageBreak())
h('2. Representations and fusion (continued)')
p('Following the distinction between feature-level and decision-level fusion [7], the two models combine modalities at different stages. Each modality is standardised using training-set statistics. The text-only, image-only and early-fusion classifiers use a 128-unit ReLU layer, dropout [8] at a rate of 0.2 and four output logits. Early fusion concatenates image and text features before classification, enabling joint feature interactions. Late fusion averages independently learned probabilities: p = w p(text) + (1-w) p(image). This decision-level approach can reduce overconfidence without learning cross-modal interactions. The known choice count masks impossible positions for every method; it never uses the correct answer.')
p('The majority-position and TF-IDF/logistic-regression baselines test label and lexical shortcuts. TF-IDF uses training-fitted unigrams/bigrams, 30,000 maximum features, minimum document frequency two and sublinear counts. Validation negative log-likelihood (NLL) selects C from {0.1, 1, 4}; C=0.1 wins. This differs from the proposal feasibility setting and is not a replication of its reported accuracy.')
h('3. Experimental design and main results')
p('The heads are trained with AdamW [9] using a learning rate of 0.001, decoupled weight decay of 0.001 and batches of 128. Training stops after six epochs without validation-NLL improvement of 0.0001, or at 40 epochs, restoring the best checkpoint. Seed 42 is primary; seeds 43/44 assess optimisation sensitivity on the same data. A validation grid w=0,0.05,...,1 selects 0.70/0.65/0.65. Source, configuration and model hashes are locked before one test evaluation. No test-dependent tuning follows.')
p('Accuracy measures exact answers; macro-F1 weights answer positions equally. NLL penalises assigning low probability to the correct answer. The multiclass Brier score sums squared errors across the answer probabilities. Both are proper scoring rules, expressed here as losses [10]. Table 2 reports the predeclared primary seed. Uniform guessing has expected accuracy 35.79%, accounting for varying K. The deterministic majority baseline has an extreme NLL because incorrect answers receive zero probability, clipped numerically.')
rows=[['Method','Accuracy %','Macro-F1','NLL','Brier']]
labels={'majority':'Majority A','tfidf':'TF-IDF + LR','image':'Image only','text':'Text only','early':'Early fusion','late':'Late fusion'}
for k in ['majority','tfidf','image','text','early','late']:
    v=primary.loc[k];rows.append([labels[k],f'{v.accuracy*100:.2f}',f'{v.macro_f1:.3f}',f'{v.nll:.3f}',f'{v.brier:.3f}'])
table(rows,[151,89,85,85,85])
p('Table 2. Test performance on all 1,979 retained questions, primary seed 42. Higher accuracy/F1 and lower NLL/Brier are preferred.','CaptionReport')
p('Text-only has the highest primary accuracy (43.86%). Early fusion reaches 42.40%, a -1.47 percentage-point difference; late fusion reaches 43.36%, a -0.51-point difference. Paired question bootstrap intervals (2,000 draws) are [-3.79, 0.81] and [-1.52, 0.45] points. These descriptive intervals include zero and may be optimistic when templates repeat. Late fusion improves NLL from 1.071 to 1.055 and Brier from 0.638 to 0.629, but neither scoring rule alone proves calibration.')
story.append(PageBreak())
h('4. Overfitting, robustness and failures')
figure('learning_curves',490,'Figure 1. Primary training and validation NLL. Dotted lines mark selected checkpoints. Continued fitting improves training fit while validation loss deteriorates.')
p('All primary heads select epoch two. Text train/validation accuracy is 62.97/44.76%, early fusion 63.97/42.72%, and image-only 46.64/37.75%. Figure 1 shows why the later accuracy peaks were not selected: the fixed objective is probability loss. Early fusion also has more trainable parameters (311,940 versus 246,404 for text), so this is a practical architecture comparison rather than a capacity-matched causal test.')
p('Across three fixed seeds, mean test accuracy and sample standard deviation are text 43.93 +/- 1.82%, early 42.95 +/- 0.53%, late 43.20 +/- 0.94%, and image 35.46 +/- 1.48%. These are optimisation summaries, not confidence intervals across independent datasets. The aggregate ordering does not support an accuracy benefit from fusion.')
p('Permuting images within choice-count strata reduces early accuracy from 42.40 to 41.74% and late accuracy from 43.36 to 43.20%. The small changes are consistent with limited useful visual contribution, although this distribution-shift diagnostic cannot establish causal reasoning. On 1,424 test questions with images unseen in training, text/early/late accuracy is 39.26/37.43/38.83%. On 1,604 unseen-text questions it is 43.39/42.27/42.83%. Neither restricted subset is a new independent benchmark.')
p('The first two early-fusion errors in test order are retained in Appendix A. In test_00001, the 253-token engineering passage is truncated to 128 tokens, omitting the actual wind-tunnel conditions. Early fusion predicts damage detection instead of parachute steadiness; truncation is a concrete information loss, not proof of the sole error cause. In test_00002, a highlighted map region is New Hampshire, while early fusion chooses Rhode Island. Global visual features may not preserve the necessary spatial distinction. The first fusion-corrected case, test_00015, changes “true” to “false” for an ice-composition statement about the largest planet, with probability only 0.516. One success does not demonstrate table reading.')
story.append(PageBreak())
h('5. Efficiency, ethics and reflection')
p('All measurements use CPU and four threads. Across the three splits, image extraction takes 438.24 seconds and text extraction 235.07 seconds, excluding model loading and audit overhead. Text sequences are deduplicated within each split, so timings are workload-specific. Primary head fitting takes 2.61 seconds for text, 2.06 for images and 2.90 for early fusion. TF-IDF search, including vectorisation and all candidates, takes 5.74 seconds.')
table([['Method','Head parameters','Checkpoint MiB','Cached ms/question'],['Text','246,404','0.94','0.00577'],['Image','66,180','0.25','0.00204'],['Early','311,940','1.19','0.00740']],[95,135,125,140])
p('Table 3. Primary head costs. Inference is the median of 20 full-test batch passes after warm-up, excluding feature extraction and scaling. MiB = 2^20 bytes.','CaptionReport')
scaler_mb=(ROOT/'models/scalers.joblib').stat().st_size/2**20
p(f'The shared pretrained weight files occupy 493.50 MiB, separate from the small fitted heads; scalers occupy {scaler_mb:.3f} MiB. Late fusion requires both unimodal heads plus probability mixing and has no separately timed end-to-end latency here. TF-IDF text vectorisation plus inference is 0.14405 ms/question and is not directly comparable to cached neural inference. No GPU speed, energy or production-throughput claim is made.')
p('Subject imbalance limits fairness claims. Primary text accuracy is 49.10% for natural science (n=1,171), 34.95% for social science (n=764), and 59.09% for language science (n=44). The smallest group is too small for a stable ranking. Grade-one and grade-twelve test groups contain only two and five questions. Grade/subject are not protected demographic attributes, and no demographic fairness conclusion follows. Answer-position macro-F1 does not measure educational fairness.')
p('The images and educational content require upstream attribution and licence compliance [1,2]. Raw datasets, pretrained weights and private course correspondence are not published in the code repository. Public pretrained encoders may contain unknown benchmark exposure, which this audit cannot rule out. Incorrect confident answers could mislead learners; these results support an experimental comparison, not an autonomous teaching or grading system. Human checking remains necessary.')
p('Implemented safeguards address different risks: train-only fitting and the selection lock prevent direct test-driven selection; legal-choice masks eliminate impossible answers but not answer-position bias. Overlap and subgroup audits reveal limitations rather than remove them. No rebalancing or demographic mitigation was evaluated, so effectiveness is unknown. Future grouped splits and balanced sampling require fresh validation; removing repeated content would also change the benchmark population. US-centred curricula and English-only evaluation limit transfer to other classrooms. Human review reduces exposure to wrong answers but has no measured benefit here.')
p('The main lesson is that richer inputs do not automatically improve this restricted model. Lost passage context and weak map/table representations motivate validating chunked text and OCR-aware features, with a capacity-matched control. Feature caching makes repeated head fitting cheap, but new inputs still incur encoder costs; these batch timings cannot establish interactive scalability. Teachers should treat the model as a research prototype pending curriculum-specific validation. Longer-context and visual improvements remain untested proposals; no extra test-driven experiments were run.')
p('The accompanying code and README provide staged commands, pinned inputs, manifests, learning curves, model-selection records and saved predictions. An independent verifier passed 136 structural, checksum and metric checks on the recorded run. Reproduction on a fresh machine remains untested.')
p('<b>Code and README</b> <link href="https://github.com/QingyanHan/comp1013-scienceqa-2026" color="#116b65">github.com/QingyanHan/comp1013-scienceqa-2026</link>', 'SmallReport')
collecting=False
story.append(PageBreak())
h('References')
refs=[
('[1]','Lu, P. et al. (2022). Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. NeurIPS 35.','https://arxiv.org/abs/2209.09513'),
('[2]','derek-thomas. ScienceQA dataset conversion. Snapshot f18b0a70359ebfb41f658fd564208d0355b013f4. Dataset card: CC BY-SA 4.0.','https://huggingface.co/datasets/derek-thomas/ScienceQA'),
('[3]','He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR, pp. 770-778.','https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html'),
('[4]','PyTorch. Torchvision ResNet-18, IMAGENET1K_V1 model documentation. Accessed 27 October 2026.','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html'),
('[5]','Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, pp. 3982-3992.','https://aclanthology.org/D19-1410/'),
('[6]','Sentence Transformers. paraphrase-multilingual-MiniLM-L12-v2 model card. Snapshot e8f8c211226b894fcb81acc59f3b34ba3efd5f42.','https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
('[7]','Baltrusaitis, T., Ahuja, C. and Morency, L.-P. (2017). Multimodal Machine Learning: A Survey and Taxonomy. arXiv:1705.09406.','https://arxiv.org/abs/1705.09406'),
('[8]','Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15, 1929-1958.','https://www.jmlr.org/papers/v15/srivastava14a.html'),
('[9]','Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.','https://arxiv.org/abs/1711.05101'),
('[10]','Gneiting, T. and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102, 359-378.','https://doi.org/10.1198/016214506000001437')]
for n,title,url in refs:p(f'{n} {html.escape(title)} <link href="{url}" color="#116b65">Source</link>.','SmallReport')
h('Generative AI assistance')
p('OpenAI Codex assisted with experimental design, implementation, execution, analysis, and drafting the report and presentation.','SmallReport')
h('Appendix A. Deterministically selected examples')
p('Selection rule fixed before evaluation: first two early-fusion errors by stable test order and first question corrected by early fusion relative to text. Only original inputs and predictions are used. Examples reproduced from ScienceQA [1,2], CC BY-SA 4.0 as reported by the conversion card; diagrams are not generated illustrations.','SmallReport')
for e in summary['examples']:
    story.append(PageBreak())
    h(f"Appendix A. {e['id']} - {e['selection'].replace('_',' ')}")
    p(html.escape(e['question']))
    im=Image(str(ROOT/e['image_file']));scale=min(330/im.imageWidth,270/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale;story.append(im);story.append(Spacer(1,10))
    table([['Option','Text']]+[[chr(65+i),c] for i,c in enumerate(e['choices'])],[50,445])
    p(f"Ground truth: {chr(65+e['answer'])}. Text prediction: {chr(65+e['text_prediction'])}. Early prediction: {chr(65+e['early_prediction'])}.")
    if e['id']=='test_00001':
        p('Relevant original hint: Gordon put a parachute with a 1 m vent in a wind tunnel simulating 200 km per hour and observed how much it swung. The retained 128-token sequence stops before those conditions. The original question/hint has 253 tokens. The image is a contextual photograph and cannot restore the missing experimental description.','SmallReport')
    elif e['id']=='test_00002':
        p('The darker green region is New Hampshire. The text contains no hint identifying the shape. Early fusion assigns 0.389 to Rhode Island and 0.229 to the correct choice, showing a spatial-recognition failure in this instance.','SmallReport')
    else:
        p('The table lists Jupiter as the largest planet and its primary composition as gas. The correct statement label is false. Early fusion assigns 0.516 to false, a narrow margin; this single corrected prediction is not evidence of reliable OCR or numerical comparison.','SmallReport')
    p('Source: ScienceQA [1,2]. Image reproduced with its original content; aspect ratio preserved.','CaptionReport')

story.append(PageBreak())
h('Appendix B. Answer-position confusion matrices')
figure('confusion_matrices',490,'Figure B1. Counts on the same 1,979 test questions, seed 42. Rows are true positions and columns predicted positions. A-D denote option positions, not semantic science categories. Empty positions are masked separately for each question.')
p('Position imbalance is substantial: test labels A/B/C/D number 706/718/366/189, and D is available only on four-choice questions. Counts therefore mix position prevalence with question format; raw diagonals alone cannot compare difficulty. Macro-F1 in Table 2 complements accuracy by weighting all four positions equally, but neither metric establishes educational fairness. These matrices describe the fixed test predictions and were not used to select a model.','SmallReport')

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c1cccf'));canvas.line(48,40,A4[0]-48,40)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#60727d'));canvas.drawString(48,28,'COMP1013 | ScienceQA | Qingyan Han a1988836');canvas.drawRightString(A4[0]-48,28,str(doc.page));canvas.restoreState()

target=OUT/'ScienceQA_Project2_2_Report.pdf'
doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52,title='ScienceQA answer prediction: early and late fusion',author='Qingyan Han; AI assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'report_main_text.txt').write_text('\n\n'.join(body_text),encoding='utf-8')
wordcount=len(re.findall(r"\b[\w]+(?:[.'/-][\w]+)*\b",' '.join(body_text)))
pdf=fitz.open(target);render=ROOT/'private/report_preview';render.mkdir(exist_ok=True)
for i,page in enumerate(pdf):page.get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(render/f'page_{i+1:02d}.png')
audit={'main_body_words_including_tables_and_captions':wordcount,'word_limit':2000,'pages':len(pdf),'pdf_text_words_first_four_pages':sum(len(p.get_text().split()) for p in list(pdf)[:4]),'contains_repository_url':any('github.com/' in link.get('uri', '') for page in pdf for link in page.get_links())}
(R/'report_build_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
assert wordcount<=2000,wordcount

