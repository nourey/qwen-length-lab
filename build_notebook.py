"""Rebuild the self-contained Kaggle notebook after editing the source files."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['experiment.py','metrics.py','plot_results.py','predictor_system.txt',
         'prompts.jsonl','smoke_prompts.jsonl','requirements-kaggle.txt',
         'tests/test_experiment.py','README.md','CONTINUE_IN_CODEX.md',
         '.gitignore','requirements-notebook.txt','scripts/setup_kaggle.sh',
         'run_kaggle.py','tests/test_packaging.py','build_notebook.py']
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        info=zipfile.ZipInfo(name, date_time=(2026,9,16,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,(ROOT/name).read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()
sha=hashlib.sha256(buffer.getvalue()).hexdigest()
cells=[]


def cell(kind, source, hidden=False):
    source=textwrap.dedent(source).strip()+'\n'
    c={'cell_type':kind,'id':f'cell-{len(cells):02d}',
       'metadata':{'jupyter':{'source_hidden':True}} if hidden else {},
       'source':source.splitlines(keepends=True)}
    if kind=='code':
        c.update(execution_count=None,outputs=[])
    cells.append(c)


cell('markdown', '''
# Can Qwen predict its next response length?

**Kaggle · Qwen3-8B · 4-bit NF4 · no-thinking target · 48-prompt pilot**

Select **Settings → Accelerator → GPU T4 ×2**, and turn **Internet on**. Then Run All.
The notebook first checks four **separate smoke prompts**, then runs 16 development
and 32 test prompts. It saves every forecast before generating target answers.
No actual GPU results are pre-filled in this notebook.

Qwen predicts five bucket probabilities; code derives P(L>128), P(L>256), P(L>512),
P(L>1024). These are verbalized probabilities to evaluate, not calibration guarantees.
The structured state/question/decision prompt is inspired by System One; it is not Jev or JEPA.

Hardware checked September 16, 2026: [Kaggle's retirement notice](https://www.kaggle.com/discussions/product-announcements/735239)
retires P100 on September 15 and identifies T4 ×2 (16 GB each). Your allocation and quota
are checked in the session. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) documents the
non-thinking switch and compatible Transformers versions.
''')
cell('code','''
from pathlib import Path
import os, sys, json, subprocess

MODEL_SIZE = "8b"       # Set to "4b" only for a deliberate new target-model experiment.
RUN_PILOT = True         # False: stop after the four-prompt hardware/format smoke check.
GPU_INDEX = 0           # One quantized model on one T4. No automatic multi-GPU sharding.
CAP = 1536              # All four evaluated thresholds must remain below this cap.
SEED = 20260916
RUN_TAG = "v1"          # Change after any experimental change. Never mix incompatible runs.

assert Path('/kaggle').is_dir(), 'Use this notebook on Kaggle with a CUDA GPU.'
assert sys.version_info >= (3, 10), 'Use a current Kaggle Python image (Python >=3.10).'
assert MODEL_SIZE in ('8b','4b')
PACKAGE_ROOT = Path('/kaggle/working/qwen-length-lab')
RUN_ROOT = Path('/kaggle/working/qwen-length-runs')
SMOKE_DIR = RUN_ROOT / f'{RUN_TAG}-smoke-qwen3-{MODEL_SIZE}'
PILOT_DIR = RUN_ROOT / f'{RUN_TAG}-pilot-qwen3-{MODEL_SIZE}'
ENV_ROOT = Path('/kaggle/temp/qwen-length-env')
os.environ['HF_HOME'] = '/kaggle/temp/qwen-length-model-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)
''')
cell('markdown','''
## Unpack the editable experiment
The next cell carries the full source bundle, dataset and tests. It preserves modified
files by refusing to overwrite them. For editing, use the extracted scripts or the separate
download bundle and run `build_notebook.py` to rebuild this notebook.
''')
cell('code',f'''
import base64, hashlib, io, zipfile
PAYLOAD = "{payload}"
blob = base64.b64decode(PAYLOAD)
assert hashlib.sha256(blob).hexdigest() == '{sha}', 'Embedded bundle checksum mismatch'
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    SOURCE_NAMES = archive.namelist()
    for entry in archive.infolist():
        relative = Path(entry.filename)
        assert not relative.is_absolute() and '..' not in relative.parts
        target = PACKAGE_ROOT / relative
        data = archive.read(entry)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Preserving modified file: {{target}}. Use a fresh package path or rebuild the notebook.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
print(f'Experiment files: {{PACKAGE_ROOT}}')
''',hidden=True)
cell('markdown','''
## Install a pinned inference stack and check the runtime
This creates a temporary environment that can use Kaggle's CUDA-enabled PyTorch.
The installed torch version is constrained so pip cannot replace it with another build.
Model downloads are cached outside the saved-output directory. No weight download occurs
until the compatibility check below succeeds. The first model download is substantial.
''')
cell('code','''
import importlib.metadata
if not (ENV_ROOT / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(ENV_ROOT)], check=True)
PYTHON = str(ENV_ROOT / 'bin/python')
constraint = ENV_ROOT / 'torch-constraint.txt'
constraint.write_text('torch==' + importlib.metadata.version('torch') + '\\n')
subprocess.run([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check',
                '-c', str(constraint), '-r', str(PACKAGE_ROOT / 'requirements-kaggle.txt')], check=True)

def experiment(*args):
    subprocess.run([PYTHON, '-u', str(PACKAGE_ROOT / 'experiment.py'), *map(str,args)],
                   cwd=PACKAGE_ROOT, check=True)

subprocess.run([PYTHON, '-m', 'unittest', 'discover', '-s', str(PACKAGE_ROOT / 'tests'), '-v'],
               cwd=PACKAGE_ROOT, check=True)
experiment('preflight', '--model-size', MODEL_SIZE, '--device', GPU_INDEX)
''')
cell('markdown','''
## Four-prompt smoke run
This checks loading, chat templating, structured predictions, actual generation and files.
It uses prompts disjoint from the pilot. It cannot establish forecasting quality.
JSON failures remain visible; a failed smoke gate stops before spending time on the pilot.
The code resolves the model revision to an immutable SHA and saves it.
''')
cell('code','''
COMMON = ['--model-size', MODEL_SIZE, '--device', GPU_INDEX, '--cap', CAP, '--seed', SEED]
experiment('run', '--mode', 'smoke', '--out', SMOKE_DIR, *COMMON)
smoke = json.loads((SMOKE_DIR / 'metrics.json').read_text())
print(json.dumps(smoke['counts'], indent=2))
assert smoke['counts']['generation_successes'] == 4, 'Inspect smoke generations.jsonl for the first error.'
assert smoke['counts']['prediction_failures'] == 0, 'Inspect smoke predictions.jsonl for raw JSON/format failures.'
REVISION = json.loads((SMOKE_DIR / 'manifest.json').read_text())['resolved_revision']
print('Pilot will use the same immutable revision:', REVISION)
print('Median prediction / generation seconds:', smoke.get('latency_seconds'))
''')
cell('markdown','''
## Frozen 48-prompt pilot
16 development outputs fit the prior and prompt-length regression baselines; 32 test
outputs score every method. A fixed text heuristic is also included. All 48 predictions
are saved before the first of these 48 target responses is generated.

Target: same NF4 model; thinking disabled; temperature 0.7, top-p 0.8, top-k 20;
1536-token cap by default. `L` includes a terminal EOS ID if emitted. Cap hits are marked.
Expected tokens use bucket midpoints, a coarse approximation.

This is a falsifiable exploratory pilot: signal requires ≥10% lower mean threshold Brier
than the strongest paired-test baseline, at least prior-level bucket accuracy, and a
family-bootstrap interval below zero, plus adequate threshold support, ≥95% valid
forecast coverage, complete generations and ≤10% cap hits. Otherwise the report says
inconclusive or no clear signal. See the extracted README for full rules and limitations.
''')
cell('code','''
if RUN_PILOT:
    experiment('run', '--mode', 'pilot', '--out', PILOT_DIR, '--revision', REVISION, *COMMON)
    ACTIVE_DIR = PILOT_DIR
else:
    ACTIVE_DIR = SMOKE_DIR
print('Results:', ACTIVE_DIR)
''')
cell('markdown','''
## Compare forecasts against measured lengths
The table and plots use identical valid test rows for each method. Full-test baseline
metrics and uncapped sensitivity results remain in `metrics.json`. Reliability curves
with 32 prompts are noisy; each Qwen reliability point shows its sample count.
''')
cell('code','''
import csv
from IPython.display import display, Markdown, Image
report = json.loads((ACTIVE_DIR / 'metrics.json').read_text())
display(Markdown('**Verdict:** ' + report['verdict']))
print(json.dumps({'counts':report['counts'], 'reasons':report['reasons'],
                  'latency_seconds':report.get('latency_seconds'),
                  'brier_delta':report.get('paired_brier_delta_bootstrap')}, indent=2))
with (ACTIVE_DIR / 'comparison.csv').open() as f:
    comparison = list(csv.DictReader(f))
if comparison:
    fields = list(comparison[0])
    table = '| ' + ' | '.join(fields) + ' |\\n| ' + ' | '.join(['---']*len(fields)) + ' |\\n'
    table += '\\n'.join('| ' + ' | '.join(row[k] for k in fields) + ' |' for row in comparison)
    display(Markdown(table))
subprocess.run([PYTHON, str(PACKAGE_ROOT / 'plot_results.py'), str(ACTIVE_DIR)], check=True)
if (ACTIVE_DIR / 'evaluation.png').exists():
    display(Image(filename=str(ACTIVE_DIR / 'evaluation.png')))
''')
cell('markdown','''
## Save the evidence
The archive contains source, prompts, the model revision, package versions, raw forecasts,
generated token IDs, CSV/JSONL results, baseline fits, scores and plots. It contains no
model weights. Save a Kaggle version or download the archive before ending the session.
Bring it back to Codex with `CONTINUE_IN_CODEX.md` to audit the actual outcome.

To resume after interruption, rerun the same settings. Do not change existing run files.
Changed code/configuration/data needs a new run directory. A 4B fallback is a separate
target experiment and must be reported separately.
''')
cell('code','''
from IPython.display import FileLink
archive_path = Path('/kaggle/working') / f'qwen_length_results_{MODEL_SIZE}.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for run in [SMOKE_DIR] + ([PILOT_DIR] if RUN_PILOT else []):
        for path in sorted(run.rglob('*')):
            if path.is_file():
                z.write(path, f'runs/{run.name}/{path.relative_to(run)}')
    # Explicit source allowlist: never archive .git, local credentials or run caches.
    for relative in SOURCE_NAMES + ['kaggle_qwen_length.ipynb']:
        path = PACKAGE_ROOT / relative
        if path.is_file():
            z.write(path, f'source/{relative}')
display(FileLink(str(archive_path)))
print('Also available from the Kaggle Output panel:', archive_path)
''')

notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,
          'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                      'language_info':{'name':'python','version':'3.11'},
                      'kaggle':{'accelerator':'gpu','isGpuEnabled':True,'isInternetEnabled':True,
                                'language':'python','sourceType':'notebook'}}}
(ROOT/'kaggle_qwen_length.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Built self-contained kaggle_qwen_length.ipynb')
