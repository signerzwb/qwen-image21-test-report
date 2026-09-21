# Prompt rewriting without the PE model (agent rewrite)

Use this when: single shots or small batches, no spare GPU for PE, or you want the
rewritten prompt to be inspectable/editable before generation. For 50+ case batches the
vLLM PE path (SKILL.md §4) is still the workhorse (stable, parallel, ~50-260s/case, no
agent round-trips).

The official PE is a 9B model executing two written recipes (system_prompt_t2i.txt /
system_prompt_edit.txt in `assets/prompt_rewrite/prompts/`). An LLM agent that follows the
distilled rules below gets equivalent prompts in seconds, no GPU. Verified A/B on 2026-09-21.

## T2I rewrite rules

1. **Split the brief.** Fixed (must survive verbatim): every text string (copy
   character-for-character, incl. punctuation/spacing), named objects, counts, colours,
   positions, user-given ratio. Open (you decide): everything else. A 3-word brief and a
   300-word brief both become the same-size description.
2. **Ratio.** Default 3:2 landscape / 2:3 portrait; 1:1 square; 16:9 cinematic/wide;
   9:16 phone. User ratio wins. The ratio becomes the W/H generation params, never words
   in the prompt.
3. **Opening sentence** (~20 words): orientation + style + medium (photograph/poster/
   illustration/close-up/page...) + subject + background & palette. Medium noun never omitted.
4. **Inventory first.** 8-14 positional phrases that reach corners/edges/centre; list of
   every legible string in reading order.
5. **Walk the frame.** Divided regions (poster/scene): background → top band → left/centre/
   right → bottom band. Single subject (portrait/close-up): background fall-off → pose &
   placement → head/face → body & garments → held items → edges. ~1/3 of sentences open on
   the positional phrase. One paragraph.
6. **Text rendering.** For each string: where it sits + how it looks + the string in straight
   quotes in its own script + weight/colour/relative size. Distant/unreadable text →
   "blurred, too small to read" — never invent letters.
7. **Lighting gets its own sentence:** source, direction, quality, shadows/highlights.
8. **Close with one composition sentence** (balance, palette, style, mood) — exactly one.
9. **Style of the prose:** English (quoted image-text stays in its own script); present
   tense, third person, declarative "observer reporting the frame" — no "create/make sure",
   no quality boosters (masterpiece/8K/highly detailed/award-winning); hedge the uncertain
   ("appears to be", "a notebook or a tablet"); colours always with a modifier (deep navy,
   pale cream); give the material (brushed metal, matte plastic, weathered wood);
   enumerate, never summarise; people by observable surface, age as life stage not number;
   no brands unless the user named one; shadows/reflections/scale must hold together.
10. **Size:** ~20 sentences, 400-500 words.

## Edit rewrite rules

1. **Two language decisions.** (A) Descriptive prose: Chinese instruction → Chinese,
   English → English, anything else → English. (B) Text rendered into the image:
   user-specified exact text/language > dominant language of existing image text >
   instruction language. Quoted strings stay monolingual.
2. **Intent branch.** "Change this picture" (local/attribute/background/text/style edit) →
   clarify + constrain: say exactly what changes, hold everything else at input fidelity.
   "New picture from these references" (placement, composite, styled shoot, poster) →
   construct actively: design scene/lighting/composition to a professional standard.
3. **Attribute disentanglement (the governing principle).** Push the named attribute(s) to
   a strong, unmistakable degree; hold everything else to input fidelity. Two symmetric
   failures: leakage (untouched things drift) and under-editing (change not visible).
4. **Anchor on the image.** Actually look at each reference (view_image) first. Every spatial/
   tonal claim comes from what is visibly there. Describe preserved content by type +
   position + role, not by repainting its appearance; one blanket preservation clause beats
   a walk of the frame.
5. **Phrasing.** Affirmative ("保持X不变"), precise and decisive, no hedging, no ratio or
   resolution in the prose, single paragraph, no newlines.
6. **In-scene text edits.** Character by character: position (match original), style
   (match original brush/type), ink colour & texture, identical size/position/typography.
7. **Outpainting / full-body.** Fill the extension coherently (e.g. lower body: garments that
   match the upper outfit's character, plausible shoes, realistic head:body ratio). New
   ratio follows the extension direction (widen for left/right, taller for up/down).

## How to run it

1. Look at reference images (edits). Rewrite the user's one-liner per the rules above.
2. Pick W/H at the 1.5K long edge: 16:9=1536x864, 3:4=1152x1536, 1:1=1536x1536, 3:2=1536x1024.
3. `scripts\t2i.py --prompt "<rewritten>" --width W --height H --steps 30 --seed 42` or
   `scripts\edit.py --ref a.png [--ref b.png] --prompt "<rewritten>" ...` (`<image1>`/`<image2>`
   numbering follows `--ref` order).
4. Show the rewritten prompt to the user before generating when they are iterating —
   the rewrite is a plain text file, editable; the PE model's output is not.

## A/B evidence (2026-09-21)

4 cases rewritten per these rules vs the official PE prompts, identical settings
(1536x864, 30 steps, seed 42, 3080 cpu_offload):

| case | axis | result |
|---|---|---|
| ab030 | t2i 中文海报双行文字 | 优，文字全对（纸上春秋 / 第12届城市书展），与 PE 版相当 |
| ab035 | t2i 英文路牌 BEIJING → 20km | 优，「BEIJING」箭头「20km」全对，与 PE 版相当 |
| ab01 | edit 背景替换+全身补全 (身份保持) | 优，身份/服装完整保留，金字塔+馆外立面一致，与 PE 版相当 |
| ab17 | edit 招牌改字 面馆→茶馆 | 优，且胜 PE 版：skill 版成功改出「茶馆」；PE 版输出仍为「面馆」（under-editing） |

结论（2026-09-21，4 例同参 A/B）：3 项打平、1 项（场景中文字替换，最难轴）skill 版胜出——
小样本下 skill 改写 ≈ 或略优于官方 PE；大批量时 vLLM PE 的分布稳定性仍有价值。

Images: `E:\qwenimage21测试\AB_skill_pe\` (abNNN.png + abNNN.txt next to PE results in
`01_文生图\` / `02_编辑测试\`).
