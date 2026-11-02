"""Inspect truncation for the preselected illustrative examples; no model fitting."""
import json
import os
from pathlib import Path
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
config = json.loads((root / 'config.json').read_text())
model = Path(os.environ.get('SCIENCEQA_TEXT_MODEL_DIR', str(Path.home() / '.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots' / config['text_model_revision'])))
tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
tokenizer.no_truncation()
tokenizer.no_padding()
examples = json.loads((root / 'results/analysis_summary.json').read_text())['examples']
out = []
for example in examples:
    text = 'Question: ' + example['question'] + ' Hint: ' + example['hint']
    tokenizer.no_truncation()
    full = tokenizer.encode(text)
    tokenizer.enable_truncation(max_length=config['text_max_tokens'])
    truncated = tokenizer.encode(text)
    out.append({'id': example['id'], 'full_tokens_including_special': len(full.ids),
                'retained_tokens_including_special': len(truncated.ids),
                'retained_text': tokenizer.decode(truncated.ids)})
(root / 'results/selected_example_token_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
