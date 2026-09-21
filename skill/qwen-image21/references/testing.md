# Testing Qwen-Image-2.1 (method + report)

No single official benchmark covers both t2i and editing; combine axes used by the common
image-gen eval suites:

## Text-to-image (t2i) axes
- Aesthetic/photography (portraits, landscapes, film look) - HPSv2-style human-preference axes
- Style diversity (anime, oil paint, ink wash, 3D, cyberpunk, Ghibli-like...)
- Composition & multi-subject layouts - T2I-CompBench-style
- Text rendering: Chinese signs/posters, English, mixed, long text (poems on boards).
  Grade character accuracy explicitly (count wrong/garbled characters).
- Content richness: animals, food, interiors, architecture, products, sci-fi

## Editing (i2i) axes - R2I-Edit / InstructPix2Pix-style categories
- Background swap (keep identity)
- Style transfer (anime, oil, ink, cyberpunk, 3D)
- Time-of-day / season change
- Object add / remove
- Attribute change (color, etc.)
- Text/sign editing in-scene
- Multi-image reference (group photo from `<image1>` + `<image2>`)
- Viewpoint / framing change (full body from a crop)

## Practical batch layout (used for the 50+20 overnight suite)
```
E:\qwenimage21测试\
  01_文生图\NNN.png + NNN.txt          # txt = raw prompt + PE prompt + params + timing
  02_编辑测试\<case>\result.png + txt + source_1.png
  03_源图\src_*.png + txt              # inputs for edit cases
  gen_log.jsonl                        # id,w,h,seconds,peak_vram_gb,status per case
  测试报告.md
```
- 1.5K = long edge 1536 (multiple of 16). 50 t2i + 20 edit = ~4-5h on a 3080 at 30 steps.
  If the user wants 16:9 landscape, pin every output to 1536x864 (override the
  per-case `wh_ratio` / reference size - keep a `ratio_src` marker so the report
  can show which aspect the model "wanted").
- Pipeline: source images first (they feed edit cases) -> PE t2i -> PE edit -> merge -> gen_batch.
- Merge step adds `out` per record (t2i -> `01_文生图`, edit -> `02_编辑测试/<id>`).
- gen_batch writes gen_log.jsonl incrementally; PE writes jsonl after each case (crash-safe).

## Parallel PE + generation (overnight watcher pattern)
PE is the slow part on 24GB cards (~30 s/case t2i, first edit case ~8 min) while
generation on a 20GB card takes ~2-3 min/case, so overlap them: vLLM t2i-PE on
GPU0, vLLM edit-PE on GPU1, and a small watcher script on GPU2 that keeps the
diffusers pipeline loaded and generates each new record from either
`*_expanded.jsonl` as it appears (oldest-first, skip ids that already have a PNG,
append to gen_log.jsonl, exit when EXPECTED=70 done or both PE logs show
 `PE_RUN_DONE` and no new cases appear for ~60 rounds). One pipeline load total;
 peak ~17GB on the 3080.

 If the PE runs are already done and only generation is left, parallelize across GPUs:
 `scripts/gen_worker.py <cuda_dev> <id1> <id2> ...` loads the pipeline once per GPU and
 processes an explicit, disjoint id list (same resolution pin / steps / seed, appends to the
 same gen_log.jsonl, skips ids whose PNG exists). Split the remaining queue ~N/3 per card
 and launch with `Start-Process -WindowStyle Hidden ... -RedirectStandardOutput worker_gpuN.log`.
 Overnight suite: 16 remaining edit cases split 6/6/4 over 2x3090 + 3080 cut wall time ~3x.

## Report structure
1. 环境与性能: GPU, drivers, versions; per-resolution latency table (s/image, s/it, peak VRAM),
   PE s/case. 2. 测试方法: axes above + counts. 3. 文生图结果: per-case table (id, category,
   verdict 优/良/中/差, notes) + per-category summary; text cases get explicit character-level
   verdicts. 4. 编辑结果: same + identity-preservation verdicts. 5. 失败/缺陷案例 with root
   cause. 6. 结论: best settings (steps/seed/res), where PE helps most, known weaknesses.
