# -*- coding: utf-8 -*-
"""Qwen-Image-2.1 single image edit (1..N reference images).

Usage:
  python edit.py --model <model_dir> --ref a.png [--ref b.png ...] --prompt "instruction"
                 [--width 1536] [--height 1536] [--steps 30] [--seed 42] [--gpu 2] [--out out.png]

Multi-image: reference the pictures inside the prompt as <image1>, <image2>, ...
"""
import argparse, os, time
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", action="append", required=True, help="reference image, repeatable")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
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

    refs = [Image.open(p).convert("RGB") for p in args.ref]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        image=refs[0] if len(refs) == 1 else refs,
        width=args.width, height=args.height,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    out = args.out or "qwen21_edit.png"
    image.save(out)
    print(f"DONE {out} {image.size[0]}x{image.size[1]} {dt:.0f}s peak={peak:.1f}GB")


if __name__ == "__main__":
    main()
