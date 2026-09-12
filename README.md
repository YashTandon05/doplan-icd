# doPlan Composition and Decomposition (ICD)

Runs Qwen2.5-VL over ICD clips to decompose
long-horizon driving instructions into sub-instructions 

> **Keep these files together in one folder** (currently `scripts/qwen/`):
> `config.py`, `dataset.py`, `prompts.py`, `review.py`,
> `gps_utils.py`. `settings.txt` is the one
> exception — it lives at the **project root** 

## Setup

**1. Get the code and a virtual environment**
```
git clone https://github.com/YashTandon05/doplan-icd.git
cd doplan-icd
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

**2. Install torch with CUDA support (Required)**

```
nvidia-smi
```
> **On a HPC cluster:** login nodes may have no GPU at all, so
> `nvidia-smi` here might be empty or misleading. Check your cluster's
> module system (e.g. `module avail cuda`) or docs, or run this from
> inside an interactive GPU session instead.

Note the "CUDA Version" shown top-right, then install the matching build
(swap `cu126` below for whatever matches — see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the
exact command for your driver):
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Confirm it worked before moving on:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This must print `True` and your GPU's name. 

**3. Install the rest**
```
pip install -r requirements.txt
```

**4. Configure your paths**
```
cp settings.example.txt settings.txt
```
Edit `settings.txt` at the **project root**: set `CSV_PATH` and
`VIDEO_PATH` to your data. Add more video paths as needed. 
```
CSV_PATH = "C:/path/to/doplan-icd/data/icd_pairs.csv"
VIDEO_PATH = "C:/path/to/las_vegas_8/las_vegas_8, C:/path/to/pittsburgh_1/pittsburgh_1"
```

**5. Get the model weights**

Either let the script auto-download on first run (slower, needs internet
every time unless cached), or pre-download for faster/offline loading:
```
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen_model_3b
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./qwen_model_7b
```
If you pre-download, point `MODEL_NAME` in the `settings.txt` you just
created at those folders — otherwise `MODEL_SIZE` (already set to a
sensible default) picks between the two automatically:
```
MODEL_NAME = "C:/path/to/doplan-icd/scripts/qwen/qwen_model_7b"
```

> **On a HPC cluster, do this instead of the above:** if compute nodes
> can't reach the internet, letting `MODEL_SIZE` auto-download inside a
> batch job will likely just time out. Pre-download on a login node,
> while you still have internet:
> ```
> hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /scratch/$USER/qwen_model_7b
> ```
> Then add this to the `settings.txt` you already created in step 4:
> ```
> MODEL_NAME = "/scratch/$USER/qwen_model_7b"
> ```
> Multiple GPUs in one job are used automatically if you request them —
> only relevant once you move to a model too large for a single GPU.

## Running prompting

```
python prompts.py --limit 2   # smoke test
python prompts.py             # full run
```

Results are written to `RESULTS_FILE` from `settings.txt` (default
`decomposition_results.json`)

If anything goes wrong — read the `[ERROR]` line, it usually
tells you which `settings.txt` value to change.

**Reviewing results against the actual video**: `review.py` reads from
the same `RESULTS_FILE`. Run this from inside `scripts/qwen` :
```
cd scripts/qwen
python review.py       # no pair_id -> lists every available pair_id
python review.py 9     # prints instruction/prediction/ground truth, opens the video + GPS map
```

If you're not in that folder, you can use the full path instead:
```
python scripts/qwen/prompts.py --limit 2
python scripts/qwen/review.py 9
```
