# -*- coding: utf-8 -*-
"""Batch prompt-enhancer (PE) runner for Qwen-Image-2.1: one model load, many cases.

Rewrites short prompts into model-optimized long prompts using the 9B PE models
(Qwen-Image-2.1-PE-T2I / Qwen-Image-2.1-PE-I2I). Reuses official prompt_rewrite
code (pe_core + run_transformers.rewrite); the code is bundled in this skill's
assets/prompt_rewrite/.

Usage:
  python pe_batch.py --task t2i  --ckpt <PE-T2I dir> --pe-dir <assets/prompt_rewrite>
                     --system-prompt <.../system_prompt_t2i.txt>
                     --inputs prompts.jsonl [--inputs more.jsonl ...]
                     --output expanded.jsonl [--seed 42] [--gpu 2]

Input JSONL: {"id": "...", "prompt": "..."} for t2i;
             {"id": "...", "prompt": "...", "input_images": ["a.png"]} for edit.
Output JSONL: id, raw_prompt, thinking, positive_prompt, wh_ratio, ratio_follow,
             parse_ok, pe_seconds (+ task, input_images, task_type, negative_prompt).
"""
import os, sys, time, argparse
from pathlib import Path

PR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "prompt_rewrite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t2i", "edit"])
    ap.add_argument("--ckpt", required=True, help="PE model dir (PE-T2I or PE-I2I)")
    ap.add_argument("--pe-dir", default=PR_DEFAULT, help="prompt_rewrite code dir (bundled)")
    ap.add_argument("--system-prompt", default=None,
                    help="system prompt txt; REQUIRED, not shipped inside the ckpt")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--device-map", default="cuda:0", help="cuda:0 keeps the 9B PE model fully on GPU (fastest); auto spills layers to CPU when VRAM is tight")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, os.path.abspath(args.pe_dir))

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import pe_core as core
    from run_transformers import rewrite

    profile = core.get_profile(args.task)
    system_prompt = core.load_system_prompt(args.system_prompt, args.ckpt)
    cases, image_paths, meta_dirs = [], [], []
    for p in args.inputs:
        p = os.path.abspath(p)
        cs = core.load_cases(p, 0)
        cases += cs
        meta_dirs += [os.path.dirname(p)] * len(cs)
        image_paths += [core.resolve_image_paths(c, os.path.dirname(p), profile) for c in cs]

    print(f"task={profile.name} cases={len(cases)}", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, low_cpu_mem_usage=True, device_map=args.device_map).eval()
    print("model loaded on:", [str(p.device) for p in list(model.parameters())[:3]], flush=True)

    records = []
    t_total = time.time()
    for i, (case, paths) in enumerate(zip(cases, image_paths)):
        images = [core.load_image(p, profile.image_max_pixels) for p in paths]
        messages = core.build_messages(system_prompt, case["prompt"], images)
        t0 = time.time()
        thinking, answer = rewrite(
            model, processor, messages,
            max_new_tokens=profile.max_new_tokens,
            temperature=profile.temperature, top_p=profile.top_p, top_k=profile.top_k,
            presence_penalty=profile.presence_penalty, seed=args.seed)
        rec = core.build_record(case, thinking, answer, profile)
        rec["pe_seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        core.write_records(Path(args.output), records)
        print(f"[{len(records)}/{len(cases)}] {case.get('id')} parse_ok={rec.get('parse_ok')} "
              f"{rec.get('pe_seconds')}s  prompt={str(rec.get('positive_prompt'))[:60]}...", flush=True)
        if i % 5 == 4:
            torch.cuda.empty_cache()
    print(f"PE_BATCH_DONE {args.output} cases={len(records)} total={time.time()-t_total:.0f}s "
          f"avg={(time.time()-t_total)/max(len(records),1):.1f}s/case", flush=True)
    core.report_parse_failures(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

