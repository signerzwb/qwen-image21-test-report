# -*- coding: utf-8 -*-
"""Batch image generation with Qwen-Image-2.1: one pipeline load, many cases.

Input JSONL record fields:
  id             output image name (without .png)
  out            output subfolder relative to --base
  prompt         plain prompt (source images) OR
  positive_prompt  PE-rewritten prompt (takes priority when present)
  input_images   list of source image paths (edit mode; optional)
  wh_ratio       "W:H" chosen by PE (or given directly); long edge -> --long-edge
  ratio_follow   "<imageN>" -> use that input image's aspect, long edge --long-edge

Writes each PNG plus one JSONL log line per case (id, out, w, h, seconds,
peak_vram_gb, status) at <base>/gen_log.jsonl. A failing case is logged and
the batch continues.

Usage:
  python gen_batch.py --expanded expanded.jsonl --base "E:\\results" --model <model dir>
                     [--long-edge 1536] [--steps 30] [--seed 42] [--gpu 2]
"""
import os, json, time, argparse
from PIL import Image


def size_for_ratio(w, h, long_edge):
    w, h = float(w), float(h)
    if w >= h:
        W = int(long_edge)
        H = int(round(long_edge * h / w / 16)) * 16
    else:
        H = int(long_edge)
        W = int(round(long_edge * w / h / 16)) * 16
    return W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--long-edge", type=int, default=1536,
                    help="target long edge; 1536 = 1.5K, 2048 = 2K")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    with open(args.expanded, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print("GPU:", torch.cuda.get_device_name(0))
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(args.model, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    log_path = os.path.join(args.base, "gen_log.jsonl")
    ok = fail = 0
    for i, case in enumerate(cases):
        cid = str(case["id"])
        out_dir = os.path.join(args.base, case.get("out", ""))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cid}.png")
        prompt = case.get("positive_prompt") or case.get("prompt")
        imgs = case.get("input_images") or []
        if case.get("wh_ratio") and ":" in str(case["wh_ratio"]):
            a, b = (int(x) for x in str(case["wh_ratio"]).split(":"))
            W, H = size_for_ratio(a, b, args.long_edge)
            ratio_src = f"wh_ratio:{case['wh_ratio']}"
        elif case.get("ratio_follow"):
            m = case["ratio_follow"].replace("<image", "").replace(">", "").strip()
            ref = Image.open(imgs[int(m) - 1])
            W, H = size_for_ratio(ref.width, ref.height, args.long_edge)
            ratio_src = f"ratio_follow:{case['ratio_follow']}"
        else:
            W, H = args.long_edge, args.long_edge
            ratio_src = "default_1:1"
        t0 = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            image = pipe(
                prompt=prompt,
                image=[Image.open(p).convert("RGB") for p in imgs] if imgs else None,
                width=W, height=H,
                num_inference_steps=args.steps,
                generator=torch.Generator("cuda").manual_seed(args.seed),
            ).images[0]
            image.save(out_path)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            ok += 1
            status = "OK"
        except Exception as exc:  # keep the batch alive
            dt = time.time() - t0
            peak = -1.0
            fail += 1
            status = f"FAIL {type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        rec = {"id": cid, "out": out_path, "w": W, "h": H, "ratio_src": ratio_src,
               "n_images": len(imgs), "seconds": round(dt, 1), "peak_vram_gb": round(peak, 2),
               "status": status}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(cases)}] {cid} {W}x{H} {ratio_src} {status} {dt:.0f}s peak={peak:.1f}GB", flush=True)
    print(f"GEN_BATCH_DONE total={len(cases)} ok={ok} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
