# -*- coding: utf-8 -*-
"""Qwen-Image-2.1 single text-to-image.

Usage:
  python t2i.py --model <model_dir> --prompt "prompt" [--width 1536] [--height 1536]
                [--steps 30] [--seed 42] [--gpu 2] [--out out.png]
"""
import argparse, os, time
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Qwen-Image-2.1 model dir")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None, help="CUDA device index (default: all visible)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("GPU:", torch.cuda.get_device_name(0))
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(args.model, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        width=args.width, height=args.height,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    out = args.out or "qwen21_t2i.png"
    image.save(out)
    print(f"DONE {out} {image.size[0]}x{image.size[1]} {dt:.0f}s peak={peak:.1f}GB")


if __name__ == "__main__":
    main()
