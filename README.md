# Qwen-Image-2.1 Windows 部署与测试报告

测试日期: 2026-09-21(通宵批测) · 硬件: RTX 3090 24GB × 2 + RTX 3080 20GB · 系统: Windows
模型: Qwen-Image-2.1(33GB, bf16) + 官方 PE-T2I / PE-I2I(各 9B / 19GB)
正文全部图片与数据都在本仓库: `images/` 成品图(每张配图同名 txt)、`data/` 提示词与统计日志、`logs/` 运行日志、`skill/` 可复现工具包。

## 摘要

| 项目 | 结果 |
|---|---|
| 文生图 | 50 张 16:9 1.5K(1536×864),平均 103s/张,峰值显存 17.7GB,**50/50 优** |
| 图像编辑 | 20 组、9 类,平均 117s/组,**19/20 优**,1 例改字未生效(§5 / §6.3) |
| 文字还原 | 8 个专项共 30+ 字符/数字,**全部逐字正确** |
| PE vs skill 改写 | 4 例同参 A/B:3 项打平,**1 项(场景中文字替换)skill 胜出** |
| 结论 | 20GB 卡 1.5K 是甜点;2K 可行但 ~2× 慢;大批量推荐 vLLM PE,单张/小批量可用 skill 规则免卡替代 |

## 1. 测试环境

| 组件 | 配置 |
|---|---|
| GPU0 / GPU1 | RTX 3090 24GB × 2 —— vLLM 跑 PE 提示词优化(PE 完成后也参与并行出图) |
| GPU2 | RTX 3080 20GB —— diffusers 出图(bf16 + `enable_model_cpu_offload`) |
| 出图环境 venv | Python 3.12, torch 2.14.0+cu126, diffusers 0.41.0.dev0 (main), transformers 5.17.0 |
| PE 环境 venv_vllm313 | Python 3.13, vLLM 0.27.1 (aivrar windows build), torch 2.13.0+cu130 |
| 出图参数 | 30 steps, seed=42, true_cfg_scale=1.0(无负向提示词), VAE slicing/tiling |

## 2. 安装与部署

两套独立环境,详见 `skill/scripts/setup_env.ps1` / `download_models.ps1`:

**出图环境(diffusers main)**
- `QwenImage21Pipeline` 只存在于 diffusers main(0.41.0.dev0),必须用 codeload tarball 安装(Windows 上 `git clone` 会卡死)
- 需要 transformers >= 5.x(注册 9B PE 模型使用的 `qwen3_5` arch)
- 模型从 ModelScope 下载: `Qwen/Qwen-Image-2.1`(33GB)、`Qwen/Qwen-Image-2.1-PE-T2I`、`Qwen/Qwen-Image-2.1-PE-I2I`(各 19GB),共 ~70GB
- CPU 内存 >= 64GB(offload 期间 33GB 权重驻留 CPU)
- 单图: `scripts\t2i.py --prompt ... --width 1536 --height 864 --steps 30 --seed 42`

**PE 环境(vLLM, Windows)**
- 官方 vLLM 无 Windows 轮子,用社区 `aivrar/vllm-windows-build`(cp313/cp314, cu130),需要 Python 3.13
- 装完必须重装 `torch==2.13.0+cu130`(pip 默认会解析成 +cpu 轮子),外加 `triton-windows` + `flash-linear-attention`
- 9B 权重 17.66GB: **20GB 卡塞不出可用 KV cache**(max-model-len 8192 都报负 KV),所以 PE 只能放 24GB 卡(`--max-model-len 16384 --gpu-memory-utilization 0.93 --max-num-seqs 2`);20GB 卡走 transformers 路径要 ~20 分钟/条
- 必须设 `PYTHONUTF8=1`(GBK 默认 locale 会破坏 torch inductor 模板加载),并用 `CUDA_VISIBLE_DEVICES` 绑卡
- 实测: 首条 edit ~8 min(Triton JIT 预热),t2i 稳态 ~30s/条,edit 后续 ~165–315s/条

## 3. 测试管线

```
短提示词(50 t2i + 20 edit)
   |
   +--> PE 扩写: vLLM @ 3090×2 并行(t2i 在 GPU0, edit 在 GPU1) --> expanded.jsonl
   |
   +--> 出图: gen_chaser.py 守护进程,看到 expanded.jsonl 新记录即上 3080 出图(oldest-first)
        PE 全部完成后,剩余 16 组编辑按 6/6/4 拆给 GPU0/GPU1/GPU2 三卡并行(gen_worker.py)
```

- `logs/chaser_log.txt` 守护进程全程日志;`logs/worker_gpu*.log` 三卡并行;`logs/pe_*.log` PE 运行
- 生成统计 `data/gen_log.jsonl`:70 条记录(耗时 / 峰值显存 / 参数 / 状态),**70/70 OK, 0 失败**

## 4. 文生图结果(50 例)

类别: portrait 8 / animal 4 / food 3 / 室内建筑街景等 12 / 文字专项 8(028–035)/ style 8 / composition 7。
全部 16:9 1536×864,每张图旁放同名 `.txt`(原始提示词 + PE 扩写提示词 + 参数 + 耗时),逐张评审明细见 `data/测试报告.md`。代表样例:

| 胶片街拍(001) | 橘猫窗台(009) |
|---|---|
| ![001](images/01_文生图/001.png) | ![009](images/01_文生图/009.png) |

| 中文海报双行文字(030) | 黑板倒计时(032) | 英文路牌(035) |
|---|---|---|
| ![030](images/01_文生图/030.png) | ![032](images/01_文生图/032.png) | ![035](images/01_文生图/035.png) |

| 水墨孤舟(039) | 五人篮球(044) |
|---|---|
| ![039](images/01_文生图/039.png) | ![044](images/01_文生图/044.png) |

**文字还原专项(028–035)**:8 例共 30+ 字符/数字,**全部逐字正确**,无错字、乱码、多余笔画;中英混排、竖排、手写体、霓虹体均稳定。这是该模型最突出的卖点之一。

## 5. 编辑结果(20 组)

类别: 背景替换 4 / 风格迁移 6 / 时间 1 / 季节 1 / 元素添加 3 / 属性 1 / 文字改字 1 / 多参考 1 / 全身补全 1 / 移除 1。
每组独立子文件夹:输出图 + 提示词 txt + 参考图(`source_*.png`)。

| 卢浮宫背景+全身补全(01) | 动漫风格迁移(05) | 白天转夜(11) |
|---|---|---|
| ![01](images/02_编辑测试/01_bg_louvre/01_bg_louvre.png) | ![05](images/02_编辑测试/05_style_anime/05_style_anime.png) | ![11](images/02_编辑测试/11_time_night/11_time_night.png) |

| 招牌改字 面馆→茶馆(17, PE 版:改字未生效) |
|---|
| ![17](images/02_编辑测试/17_text_sign/17_text_sign.png) |

**判定:19/20 优。** 唯一未达标:`17_text_sign`——PE 扩写提示词逐字写明了位置/字体/墨色,但输出招牌仍是「面馆」(under-editing,变更未生效),且板色由黑板白字反转为白板黑字;灯笼、OPEN 牌、视角等场景元素保留良好。对照:同一指令用 skill 规则改写在 A/B 中成功改出「茶馆」(§6.3)。

## 6. PE 的使用与 skill 替代

### 6.1 PE 逻辑(自两份官方 system prompt 蒸馏)

PE = 两个 9B Qwen3.5(thinking 开启)各执行一份写死的食谱,原文见 `skill/assets/prompt_rewrite/prompts/`:

**T2I 食谱(8 步)**
1. 拆 brief 为「必须逐字保留」(所有文字串、命名物体、数量、颜色、位置、用户指定画幅)与「自由脑补」(其余);3 词 brief 与 300 词 brief 都扩到同等长度
2. 画幅:默认 3:2 / 1:1 / 16:9 / 9:16,用户指定优先;只进生成参数,不进正文
3. 开场句:方向 + 风格 + 媒介 + 主体 + 背景色板
4. 先列 8–14 个覆盖边角与中心的位置词 + 要渲染文字的全清单(按阅读顺序)
5. 「走帧」分区(海报:背景 → 顶部横带 → 左/中/右 → 底部;人像:背景 → 姿态 → 头脸 → 身体 → 手持 → 边缘),约 1/3 句子以位置短语开头
6. 文字渲染:引号内原文逐字 + 字体/颜色/相对尺寸;远处小字写 blurred,不造字
7. 光线单独一句(光源、方向、质量、阴影)
8. 结尾一句构图总结;风格:英文观察者视角陈述句、无 masterpiece/8K 类加分词、颜色必带修饰词、写材质、不确定处 hedge、~400–500 词

**Edit 食谱(5 条)**
1. 双层语言决策:描述性 prose 跟用户语言;图中要渲染的文字按「用户指定 > 原图主语言 > 指令语言」,引号内单语
2. 意图分支:局部修改 = 澄清 + 锁定其余;参考图出新场景 = 主动设计构图/光线
3. 核心原则 = 属性解耦:目标属性推到强而明确,其余锁输入图;两种对称失败:leakage(无关处漂移)/ under-editing(变更未生效)
4. 先看参考图锚定;保留项按「类型+位置+角色」点名 + 一条总保持条款
5. 肯定式表述、不 hedge、正文不写分辨率

### 6.2 skill 替代(免卡)

把两份食谱蒸馏进 `skill/references/prompt-rewrite.md`(T2I 10 条 + Edit 7 条 + 用法):agent 照规则改写后直接进 `t2i.py` / `edit.py`——**零显存、秒级、改写结果是可检查可手改的文本**。

| 场景 | 建议 |
|---|---|
| 单张 / 小批量 / 没有空闲 GPU / 想逐字审提示词 | skill 改写(本仓库 §6.3 的 4 例即按此流程产出) |
| 50+ 大批量 | vLLM PE:稳定并行、无 agent 编排;按出图模型分布微调,长批量风格稳定性略优 |

### 6.3 A/B 对照(4 例,1536×864 / 30 steps / seed 42,参数完全相同)

| 用例 | 轴 | 结果 |
|---|---|---|
| ab030 | t2i 中文海报双行文字 | 打平,文字全对 |
| ab035 | t2i 英文路牌 BEIJING → 20km | 打平,全对 |
| ab01 | edit 背景替换 + 全身补全(身份保持) | 打平,身份/服装完整保留 |
| ab17 | edit 招牌改字 面馆→茶馆 | **skill 胜**:skill 版改出「茶馆」;PE 版输出仍为「面馆」 |

| PE 版 030 | skill 版 ab030 |
|---|---|
| ![pe030](images/01_文生图/030.png) | ![ab030](images/AB_skill_pe/ab030.png) |

| PE 版 17(仍是「面馆」) | skill 版 ab17(「茶馆」) |
|---|---|
| ![pe17](images/02_编辑测试/17_text_sign/17_text_sign.png) | ![ab17](images/AB_skill_pe/ab17.png) |

小样本下 skill 改写 ≈ 或略优于官方 PE;PE 的价值集中在大批量稳定性与零人工干预。

## 7. 结论与建议

1. **推荐配置**:20GB 卡 bf16 + cpu_offload + VAE slicing/tiling,30 steps,1.5K 长边;16:9 1536×864 约 100–135s/张、峰值 ~17.5GB;2K 能跑但 ~2× 慢
2. **PE 值得上**:逐字符拆解直接决定文字还原率;编辑类「不得改变」清单让身份/场景保真更稳;但并非无懈可击(17 号 under-editing),单张/小批量可用 skill 改写兜底
3. **多卡分工**:PE 吃 24GB 级卡(20GB 卡 vLLM 塞不下 KV),出图 20GB 就够;2×3090 出词 + 1×3080 出图互不抢卡,PE 完成后剩余出图可再按卡拆分并行(3× 墙钟加速)
4. **已知短板**:双参考图编辑显存逼近 20GB 上限(20.1GB)、耗时 ×3.7;强风格迁移(油画/水墨)会牺牲部分身份相似度;强制画幅与模型推荐画幅冲突时以用户指令为准

## 8. 复现指南(给另一个智能体)

`skill/` 目录即 Codex skill `qwen-image21`(18 个文件,自包含)——读完后即可完成 安装 → 下载 → 出图 → 编辑 → 批量测试 → 报告:

```
skill/
├── SKILL.md                        # 主入口:安装 / 单图 / 编辑 / PE / vLLM / 坑清单
├── scripts/
│   ├── setup_env.ps1               # venv + torch cu126 + diffusers main
│   ├── download_models.ps1         # ModelScope 下载 3 个模型(~70GB)
│   ├── t2i.py / edit.py            # 单张文生图 / 单张编辑
│   ├── pe_batch.py / gen_batch.py  # 批量 PE 扩写 / 批量出图
│   └── gen_chaser.py / gen_worker.py  # 通宵守护(oldest-first) / 多卡并行 worker
├── references/
│   ├── testing.md                  # 测试集与报告写法(50 t2i + 20 edit 约定)
│   └── prompt-rewrite.md           # PE 食谱蒸馏(skill 改写规则, A/B 验证 ≈ PE)
└── assets/prompt_rewrite/          # 官方 PE 代码(pe_core.py / run_vllm.py) + 两份 system prompt
```

## 9. 附录:仓库结构

```
images/
├── 01_文生图/           # 50 张 PNG + 50 个同名 TXT(原始提示词 + PE 扩写 + 参数 + 耗时)
├── 02_编辑测试/         # 20 个子文件夹(输出图 + TXT + 参考图 source_*.png)
├── 03_源图/             # 6 张编辑源图
└── AB_skill_pe/         # 4 张 A/B 图 + 4 个 TXT(skill 改写提示词)
data/
├── 测试报告.md           # 详细测试报告(环境 / 管线 / 性能表 / 逐例评审)
├── gen_log.jsonl        # 70 条生成日志(耗时 / 峰值显存 / 参数 / 状态)
├── t2i_prompts.jsonl / t2i_expanded.jsonl    # 50 条原始短提示词 / PE 扩写结果
├── edit_cases.jsonl / edit_expanded.jsonl    # 20 条原始编辑指令(含参考图) / PE 扩写结果
└── ab_cases.jsonl       # 4 条 skill 改写提示词(A/B)
logs/                    # PE vLLM 运行、chaser 守护、三卡并行 worker、A/B 运行原始日志
skill/                   # 见 §8
```
