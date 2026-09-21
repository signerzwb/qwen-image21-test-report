# -*- coding: utf-8 -*-
"""Generate a fixed list of cases on one GPU (companion to gen_chaser.py).

Usage: python gen_worker.py <cuda_device> <id1> <id2> ...
Skips ids whose output PNG already exists; appends records to gen_log.jsonl.
"""
import os, json, time, sys

WS = r"C:\Users\Administrator\Documents\ChatGPT\New project"
BASE = r"E:\qwenimage21测试"
MODEL_DIR = os.path.join(WS, "models", "Qwen-Image-2.1")


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    dev = sys.argv[1]
    ids = sys.argv[2:]
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    import torch
    from PIL import Image
    print(f"WORKER dev={dev} ids={ids} gpu={torch.cuda.get_device_name(0)}", flush=True)
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    cases = {}
    for p in ("t2i_expanded.jsonl", "edit_expanded.jsonl"):
        for r in load(os.path.join(WS, "test_assets", p)):
            cases[str(r["id"])] = r

    log_path = os.path.join(BASE, "gen_log.jsonl")
    done = 0
    for cid in ids:
        case = cases.get(cid)
        if case is None:
            print(f"SKIP {cid} not found in expanded jsonl", flush=True)
            continue
        is_edit = cid in {str(r["id"]) for r in load(os.path.join(WS, "test_assets", "edit_expanded.jsonl"))}
        out_dir = os.path.join(BASE, f"02_编辑测试/{cid}" if is_edit else "01_文生图")
        out_path = os.path.join(out_dir, f"{cid}.png")
        if os.path.exists(out_path):
            print(f"SKIP {cid} exists", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        prompt = case.get("positive_prompt") or case.get("prompt") or ""
        imgs = case.get("input_images") or []
        W, H = 1536, 864  # FORCE_169 per user request
        pe_ratio = case.get("wh_ratio") or case.get("ratio_follow") or "none"
        t0 = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            image = pipe(
                prompt=prompt,
                image=[Image.open(p).convert("RGB") for p in imgs] if imgs else None,
                width=W, height=H,
                num_inference_steps=30,
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]
            image.save(out_path)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            status = "OK"
        except Exception as exc:
            peak = -1.0
            status = f"FAIL {type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        done += 1
        rec = {"id": cid, "out": out_path, "w": W, "h": H, "ratio_src": f"force_16:9(pe={pe_ratio})",
               "n_images": len(imgs), "seconds": round(time.time() - t0, 1),
               "peak_vram_gb": round(peak, 2), "status": status, "gpu": f"cuda:{dev}"}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{done}/{len(ids)}] {cid} {W}x{H} {status} {time.time() - t0:.0f}s peak={peak:.1f}GB", flush=True)
    print(f"WORKER_DONE dev={dev} done={done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
