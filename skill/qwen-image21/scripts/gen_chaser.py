# -*- coding: utf-8 -*-
"""Watch PE-expanded JSONL files and generate images as new cases appear.

Runs on a dedicated GPU (default 2). The Qwen-Image-2.1 pipeline stays loaded;
each loop picks the oldest not-yet-generated case, renders it, and appends a
record to E:\\qwenimage21测试\\gen_log.jsonl. Exits when all expected cases are
done, or when both PE runs have printed PE_RUN_DONE and nothing is pending.
"""
import os, json, time, sys, re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

WS = r"C:\Users\Administrator\Documents\ChatGPT\New project"
BASE = r"E:\qwenimage21测试"
MODEL_DIR = os.path.join(WS, "models", "Qwen-Image-2.1")
LONG_EDGE = 1536
EXPECTED = 70  # 50 t2i + 20 edit
FORCE_169 = True  # user request: all outputs 16:9 landscape (1536x864)
PE_LOGS = [os.path.join(WS, "pe_t2i_full.log"), os.path.join(WS, "pe_edit_full.log")]


def load(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def index(p):
    return {str(r["id"]): r for r in load(p)}


def size_for_ratio(w, h, long_edge=LONG_EDGE):
    w, h = float(w), float(h)
    if w >= h:
        W = int(long_edge)
        H = int(round(long_edge * h / w / 16)) * 16
    else:
        H = int(long_edge)
        W = int(round(long_edge * w / h / 16)) * 16
    return W, H


def build_cases():
    t2i_exp = load(os.path.join(WS, "test_assets", "t2i_expanded.jsonl"))
    edit_exp = load(os.path.join(WS, "test_assets", "edit_expanded.jsonl"))
    t2i_cat = index(os.path.join(WS, "test_assets", "t2i_prompts.jsonl"))
    edit_cat = index(os.path.join(WS, "test_assets", "edit_cases.jsonl"))
    out = []
    for rec in t2i_exp:
        rec = dict(rec)
        rec["out"] = "01_文生图"
        rec["category"] = t2i_cat.get(str(rec["id"]), {}).get("category", "")
        out.append(rec)
    for rec in edit_exp:
        rec = dict(rec)
        rec["out"] = f"02_编辑测试/{rec['id']}"
        rec["category"] = edit_cat.get(str(rec["id"]), {}).get("category", "")
        out.append(rec)
    return out


def pe_finished():
    done = 0
    for p in PE_LOGS:
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                if "PE_RUN_DONE" in f.read():
                    done += 1
    return done == len(PE_LOGS)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    import torch
    from PIL import Image
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    log_path = os.path.join(BASE, "gen_log.jsonl")
    processed = 0
    idle_rounds = 0
    while processed < EXPECTED:
        cases = build_cases()
        new = [c for c in cases
               if not os.path.exists(os.path.join(BASE, c["out"], f"{c['id']}.png"))]
        if not new:
            time.sleep(30)
            idle_rounds += 1
            if idle_rounds >= 60 and pe_finished():
                print(f"CHASER_STOP no pending cases and both PE runs finished "
                      f"(processed={processed})", flush=True)
                break
            continue
        case = new[0]
        cid = str(case["id"])
        out_dir = os.path.join(BASE, case["out"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cid}.png")
        prompt = case.get("positive_prompt") or case.get("prompt") or ""
        imgs = case.get("input_images") or []
        if FORCE_169:
            W, H = 1536, 864
            pe_ratio = case.get("wh_ratio") or case.get("ratio_follow") or "none"
            ratio_src = f"force_16:9(pe={pe_ratio})"
        elif case.get("wh_ratio") and ":" in str(case["wh_ratio"]):
            a, b = (int(x) for x in str(case["wh_ratio"]).split(":"))
            W, H = size_for_ratio(a, b)
            ratio_src = f"wh_ratio:{case['wh_ratio']}"
        elif case.get("ratio_follow"):
            m = case["ratio_follow"].replace("<image", "").replace(">", "").strip()
            ref = Image.open(imgs[int(m) - 1])
            W, H = size_for_ratio(ref.width, ref.height)
            ratio_src = f"ratio_follow:{case['ratio_follow']}"
        else:
            W, H = LONG_EDGE, LONG_EDGE
            ratio_src = "default_1:1"
        t0 = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            image = pipe(
                prompt=prompt,
                image=[Image.open(p).convert("RGB") for p in imgs] if imgs else None,
                width=W, height=H,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            image.save(out_path)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            status = "OK"
        except Exception as exc:
            peak = -1.0
            status = f"FAIL {type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        processed += 1
        idle_rounds = 0
        rec = {"id": cid, "out": out_path, "w": W, "h": H, "ratio_src": ratio_src,
               "n_images": len(imgs), "seconds": round(time.time() - t0, 1),
               "peak_vram_gb": round(peak, 2), "status": status}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{processed}/{EXPECTED}] {cid} {W}x{H} {ratio_src} {status} "
              f"{time.time() - t0:.0f}s peak={peak:.1f}GB", flush=True)
    print(f"CHASER_DONE processed={processed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
