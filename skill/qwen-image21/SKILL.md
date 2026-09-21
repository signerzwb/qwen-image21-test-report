---
name: qwen-image21
description: Install, download, and run Qwen-Image-2.1 (text-to-image + image editing, 1K-2K, strong Chinese/English text rendering) with its 9B prompt-enhancer (PE) models on a Windows GPU box via diffusers. Use when the user wants to generate or edit images with Qwen-Image-2.1, or batch-run/evaluate it.
---

# Qwen-Image-2.1 (Windows, diffusers)

Run Qwen-Image-2.1 locally: text-to-image and 1..N-image editing at 1024-2048px.
Verified working on Windows + RTX 3080 20GB (CUDA 12.6, bf16, `enable_model_cpu_offload`).
Peak VRAM ~16.4GB at 1536px, ~17.3GB at 16:9 1.5K, 2K fits on 20GB but is ~2x slower.
CPU RAM should be >=64GB (33GB model weights live on CPU during offload).

## Layout

- `scripts/setup_env.ps1` - create `./venv` with pinned versions + diffusers from main
- `scripts/download_models.ps1` - ModelScope download of all 3 models into `./models/`
- `scripts/t2i.py` / `scripts/edit.py` - single image generation / edit
- `scripts/pe_batch.py` / `scripts/gen_batch.py` - batch PE rewriting / batch generation
- `scripts/gen_chaser.py` / `scripts/gen_worker.py` - overnight watcher (oldest-first, exits on PE_RUN_DONE) / explicit-id parallel worker for filling the gen queue on several GPUs
- `assets/prompt_rewrite/` - official PE code (`pe_core.py`, `run_transformers.py`) + system prompts
- `references/prompt-rewrite.md` - agent-side prompt-rewrite rules (distilled PE system prompts), A/B-verified ≈ PE; for single/small batches or no free GPU
- `references/testing.md` - how to batch-test t2i and edit, and write a report

## 1. Install (PowerShell, in the working directory)

```powershell
powershell -File <skill>\scripts\setup_env.ps1     # venv + torch cu126 + transformers 5.17 + diffusers main
powershell -File <skill>\scripts\download_models.ps1   # ~70GB total into ./models
```

Key pinning facts:
- `QwenImage21Pipeline` exists ONLY in diffusers main (0.41.0.dev0). Install via codeload tarball
  (`https://codeload.github.com/huggingface/diffusers/tar.gz/refs/heads/main`), never `git clone` (stalls on Windows).
- transformers >=5.x is required (registers the `qwen3_5` arch used by the PE models).
- Modelscope is the fast source in mainland China: `Qwen/Qwen-Image-2.1` (33GB),
  `Qwen/Qwen-Image-2.1-PE-T2I` and `Qwen/Qwen-Image-2.1-PE-I2I` (~19GB each).
- Completion markers: `.msc_cache` folder or `_____temp` absence inside each model dir.

## 2. Generate / edit (single)

```powershell
.\venv\Scripts\python.exe -X utf8 <skill>\scripts\t2i.py --model models\Qwen-Image-2.1 `
  --prompt "胶片写真，中国女性，街拍" --width 1536 --height 1536 --steps 30 --seed 42 --out out.png

.\venv\Scripts\python.exe -X utf8 <skill>\scripts\edit.py --model models\Qwen-Image-2.1 `
  --ref portrait.png --prompt "把背景换成卢浮宫，16:9 全身照，保持人物不变" `
  --width 1536 --height 864 --out edit.png
```

- Always pass explicit `--width/--height` (multiples of 16). Default target: long edge 1536 (1.5K).
  16:9 = 1536x864; 3:4 = 1152x1536; 1:1 = 1536x1536. Use `--long-edge 2048` in batch mode for 2K.
- Multiple references: repeat `--ref`; refer to them in the prompt as `<image1>`, `<image2>`.
- Use `--gpu N` to pin a specific GPU (nvidia-smi index). `true_cfg_scale=1.0` by default -
  there is NO negative prompt; all control is in the positive prompt.
- Baseline speed (3080, cpu_offload, 30 steps): 1024x1024 ~100-135s; 1536x864 (16:9) ~101-134s,
  peak 16.7-17.5GB (measured over a 70-case suite); 1536px square/3:4 ~180-250s; 2048x2048 ~514s.
- Always use `dtype=` (not deprecated `torch_dtype=`); keep VAE slicing+tiling on for >=1536px.

## 3. Prompt enhancer (PE) - use for anything but a trivial one-line prompt

The PE models (9B Qwen3.5 hybrids, "thinking" enabled) rewrite a short user prompt into a
long model-optimized prompt, and also pick an aspect ratio. Strongly recommended whenever the
prompt has multiple constraints, text to render (Chinese or English), or a specific composition.
For edit tasks use PE-I2I; its output may reference `<image1>` etc. and may set `ratio_follow`
to keep the reference image's aspect.

```powershell
# 1) rewrite prompts (one model load for the whole batch; ~20-60s/case)
.\venv\Scripts\python.exe -X utf8 <skill>\scripts\pe_batch.py --task t2i `
  --ckpt models\Qwen-Image-2.1-PE-T2I --pe-dir <skill>\assets\prompt_rewrite `
  --system-prompt <skill>\assets\prompt_rewrite\prompts\system_prompt_t2i.txt `
  --inputs prompts.jsonl --output expanded.jsonl --gpu 2

# 2) generate with the rewritten prompts
.\venv\Scripts\python.exe -X utf8 <skill>\scripts\gen_batch.py --model models\Qwen-Image-2.1 `
  --expanded expanded.jsonl --base "E:\results" --long-edge 1536 --steps 30 --seed 42 --gpu 2
```

- `--system-prompt` is REQUIRED (not shipped inside the PE ckpt dirs).
- t2i profile uses presence_penalty 1.5 / max_new 16256; edit uses 0 / 24000 (handled by `--task`).
- Output records: `positive_prompt` (what gen_batch sends to the model), `thinking`, `wh_ratio`,
  `ratio_follow`, `parse_ok`. If `parse_ok` is false, fall back to the raw prompt.
- PE needs ~18GB VRAM alone; do NOT run it concurrently with image generation on the same GPU.
- PE inputs: t2i = `{"id","prompt"}`; edit = `{"id","prompt","input_images":["a.png"]}`.
  Reference images are resized to <=1MP before PE sees them.

- **PE alternative (agent/skill rewrite, no GPU)**: for single shots, small batches, or no free GPU,
  the agent can rewrite prompts itself following `references/prompt-rewrite.md` (distilled from the two
  PE system prompts) and generate with `t2i.py` / `edit.py` as usual: seconds, zero VRAM, and the
  rewritten prompt is inspectable, editable text. A/B on 2026-09-21 (4 cases, same params): tied PE on
  3 axes (zh text, en text, identity preservation) and beat it on in-scene text replacement (PE
  under-edited and kept the original text). For 50+ case batches prefer the vLLM PE path (§4).

## 4. PE via vLLM (fast path, Windows)

The transformers path above runs the 9B PE at only ~5.5 tok/s on a 3080 (hybrid
linear-attention decode is kernel-bound). For batch work (dozens of prompts) use
vLLM. Official vLLM has no Windows wheels; use the community `aivrar/vllm-windows-build`
prebuilt wheels (cp313/cp314, cu130):

```powershell
# 1) Python 3.13 (not 3.12!) - the wheels are cp313/cp314 only
winget install Python.Python.3.13 --accept-source-agreements --accept-package-agreements
& "C:\Users\...\Python313\python.exe" -m venv venv_vllm313

# 2) the wheel (needs an exact wheel-style filename for pip):
Invoke-WebRequest "https://github.com/aivrar/vllm-windows-build/releases/download/v0.27.1-win-cu130/vllm-0.27.1-cp313-cp313-win_amd64.whl" -OutFile vllm-0.27.1-cp313-cp313-win_amd64.whl
& venv_vllm313\Scripts\pip.exe install vllm-0.27.1-cp313-cp313-win_amd64.whl

# 3) CRITICAL: pip resolves the pinned torch to the +cpu wheel. Reinstall CUDA:
& venv_vllm313\Scripts\pip.exe install "torch==2.13.0+cu130" "torchaudio==2.11.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130
& venv_vllm313\Scripts\pip.exe install triton-windows flash-linear-attention
```

On a 20GB GPU the 17.66GB weights leave NO usable KV cache: even
`--max-model-len 8192 --max-num-seqs 1` reports negative KV memory on a 3080 20GB,
so **vLLM PE does not fit on 20GB cards** - use the transformers path there
(~20 min/case) or a 24GB+ GPU. On a 24GB card (RTX 3090, verified):
`--max-model-len 16384 --gpu-memory-utilization 0.93 --max-num-seqs 2`, and
cudagraphs are fine (drop `--enforce-eager`, ~90s one-time inductor compile).
Measured on a 3090: t2i ~30s/case steady state; the first edit case takes ~8 min
(Triton JIT warmup), later ones much faster. If you get "No available memory for
the cache blocks", lower max-model-len / max-num-seqs.

Run it with `assets/prompt_rewrite/run_vllm.py` (patched: drop the `swap_space=`
arg removed in vLLM >=0.27, chunked batch with `--chunk N`, per-case progress
`[n/m] id=... parse_ok=... elapsed=...s`, and resume-skip of ids already in
`--output`). The run needs `PYTHONUTF8=1` (GBK default locale breaks
torch._inductor template loading) and `CUDA_VISIBLE_DEVICES` pinned:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; $env:PYTHONUTF8="1"
& venv_vllm313\Scripts\python.exe -X utf8 <skill>\assets\prompt_rewrite\run_vllm.py `
  --task t2i --ckpt models\Qwen-Image-2.1-PE-T2I `
  --input prompts.jsonl --output expanded.jsonl `
  --system-prompt <skill>\assets\prompt_rewrite\prompts\system_prompt_t2i.txt `
  --max-model-len 16384 --gpu-memory-utilization 0.93 --max-num-seqs 2 --chunk 2
```

Output records are identical to the transformers path (except no `pe_seconds`).
Run multiple PE tasks in parallel on separate GPUs (e.g. t2i on GPU0, edit on
GPU1) while image generation runs on a third GPU - see `references/testing.md`
for the watcher pattern.

## 5. Gotchas

- Windows/PowerShell: no heredocs (`@'...'@` here-strings or temp .py files); no `head` (use
  `Select-Object -Last N`); quote paths with spaces.
- OOM during a batch is caught and logged (`status=FAIL ...`); the batch continues. 2048px +
  2 reference images can exceed 20GB - drop to 1536 or fewer refs.
- The PE model and the image model must not share one 20GB GPU at the same time.
- `CUDA_VISIBLE_DEVICES` is set per script via `--gpu` (or `os.environ.setdefault` in older copies);
  never disturb other GPUs that host unrelated jobs.
- If a modelscope dir looks incomplete, delete `._____temp` folders (via `python -c "import shutil;shutil.rmtree(...)"`
  if PowerShell refuses dot-named items) and re-run the download - it resumes.

## 6. Testing & reports

See `references/testing.md` for the tested test-suite layout (50 t2i + 20 edit cases,
per-case prompt sidecar txt, performance log, report structure).
