# -*- coding: utf-8 -*-
"""Merge t2i_expanded + edit_expanded into test_assets/gen_all.jsonl with
per-record out dir and category (input for finalize_outputs.py)."""
import json, os
WS = r"C:\Users\Administrator\Documents\ChatGPT\New project"


def load(p):
    with open(os.path.join(WS, p), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def idx(p):
    return {str(r["id"]): r for r in load(p)}


t2i_cat = idx("test_assets/t2i_prompts.jsonl")
edit_cat = idx("test_assets/edit_cases.jsonl")
out = []
for r in load("test_assets/t2i_expanded.jsonl"):
    r = dict(r)
    r["out"] = "01_文生图"
    r["category"] = t2i_cat.get(str(r["id"]), {}).get("category", "")
    out.append(r)
for r in load("test_assets/edit_expanded.jsonl"):
    r = dict(r)
    r["out"] = "02_编辑测试/" + str(r["id"])
    r["category"] = edit_cat.get(str(r["id"]), {}).get("category", "")
    out.append(r)
path = os.path.join(WS, "test_assets", "gen_all.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("gen_all records:", len(out))
