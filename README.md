# CARER-Net 中文复现代码

对应论文：**CARER-Net：面向眼底血管精细分割的有界残差颜色适配方法**。

主配置 **CARER-Core** 在灰度分割主干前插入颜色细节适配器，以 RGB 和绿色通道局部对比生成灰度残差。适配器新增 **1,201** 个参数，保留灰度旁路，输出层零初始化。可选 **CARER-Refine** 在输出端加入误差监督双分支精修器。

```text
眼底 RGB ──→ 颜色与局部对比特征 ──→ 灰度残差 Δx
                    灰度旁路 xg + 0.1 × tanh(Δx)
                                  ↓
                         GN 残差注意力 U-Net
                                  ↓
                             血管分割结果
```

本仓库是独立的中文代码与数值记录发布版。英文论文仓库 [`Ironman-creator/CARER-Net`](https://github.com/Ironman-creator/CARER-Net) 独立维护，其文件、分支和提交未因本次发布而修改。

## 快速开始：无需训练核对论文表格

在 Python 3.10 或更高版本的隔离环境中运行：

```bash
python -m pip install -r requirements-statistics.txt
python scripts/verify_release.py
python scripts/reproduce_tables.py
python component_ablation/analyze_published.py
```

这些命令只读取随仓库发布的数值文件，结果写入 `artifacts/`，不需要 GPU、数据集或模型权重。

- `reproduce_tables.py` 从逐种子、逐图结果核对主实验均值、标准差及配对差值，整理中文表2–6；历史置信区间取自原统计输出。
- `analyze_published.py` 从9组×200张图像的结果**重新计算**组件消融的均值、种子间标准差，以及4项比较的95%和98.75%区间，并逐项核对发布值。

## 重新训练与测试

先从数据原始提供方取得数据，按[数据准备说明](docs/数据准备.md)整理；然后安装与驱动匹配的 PyTorch 和 `requirements.txt`。报告中的训练环境为 PyTorch `2.11.0+cu128`。完整操作见[训练复现](docs/训练复现.md)。

先生成计划，检查固定划分、参数和输出位置：

```bash
python scripts/run_main_experiments.py --datasets FIVES --seeds 42 123 3407 --plan-out artifacts/fives_plan.json
```

确认后加 `--execute` 执行。四数据集和表2比较模型可使用：

```bash
python scripts/run_main_experiments.py --datasets FIVES DRIVE STARE CHASE --all-baselines --plan-out artifacts/full_plan.json
```

默认命令不会启动训练。所有新训练结果写入独立 `artifacts/main_reproduction/`，不会覆盖 `results/` 中的历史记录。

## 实验配置

| 配置 | 代码名称 | 日程 |
|---|---|---|
| 原始 GN 主干 | `res_attention_unet_gn` | FIVES/DRIVE/CHASE最多50轮；STARE最多100轮 |
| 轮数匹配灰度对照 | `gray_rgb_control_30_unet` | 30轮末端更新 |
| 主配置 CARER-Core | `color_detail_only_unet` | 5轮适配器预热＋25轮适配器与末端联合更新 |
| 可选 CARER-Refine | `carer_unet` | 同上，增加误差监督精修器 |

固定种子为42、123、3407，输入512×512，批量大小1。模型按验证集soft Dice选点，测试使用固定阈值0.5及 `probability > 0.5` 规则，视野由PIL灰度图大于5/255确定。完整参数及命令见文档。

## 新增组件消融

三个方案具有相同的1,201个新增参数，区别仅为是否使用 `tanh`、输出卷积是否零初始化。

| 方案 | Dice均值±种子样本标准差 | 细血管中心线召回均值 |
|---|---|---|
| 完整 Core | 0.898347 ± 0.001756 | 0.792828 |
| 取消 `tanh`、保留0.1系数 | 0.898415 ± 0.001760 | 0.793352 |
| 输出层默认随机初始化 | 0.897238 ± 0.002079 | 0.790552 |

零初始化在插入时保持原主干预测起点，并在该实验日程下改善Dice和细血管中心线召回。取消 `tanh` 后的性能差异区间跨零；保留它的作用是限制修正幅度，主配置仍为完整 Core。该实验与历史主实验分开，历史FIVES Core Dice仍为0.898350。

组件代码、原始记录、固定权重SHA和严格训练流程见[组件消融说明](component_ablation/README.md)。数值复算不需要权重；重跑原封存组件实验需要对应的三个历史GN源检查点。

## 目录

```text
models/                   主干、Core、Refine及比较模型
train.py / test.py        基础训练与固定阈值评价
train_carer_ablation.py   轮数匹配对照及两个CARER配置
scripts/                  中文命令计划、主表核对、发布完整性检查
analysis/                 血管骨架、管径、分支点、边界评价
splits/main/              四数据集×三个种子的固定训练/验证成员
results/                  历史主实验及逐图数值记录
component_ablation/       三方案组件代码与9组原始记录
docs/                     数据、训练、表格来源说明
```

## 数据、权重与结果来源

仓库不分发眼底照片、人工标注或训练权重。数据由各原始提供方发布，许可不由本仓库的MIT许可证替代。传统三个数据集使用历史gamma预处理灰度副本，FIVES使用原生RGB；替换成其他输入会改变实验条件。详见[数据准备](docs/数据准备.md)。

主代码来自完成四数据集实验的冻结训练实现；组件实验保留完成本机实验的独立快照。原始统计文件按字节保留，表格映射和统计口径见[实验结果索引](docs/实验结果索引.md)。所有发布文件的SHA-256列于 `RELEASE_MANIFEST.json`。

## 引用与联系

仓库引用信息见 [`CITATION.cff`](CITATION.cff)。本仓库为中文研究代码发布，不表示论文已经正式发表。

Zijian Chen，Guangzhou Huanan Business College。  
联系邮箱：mrchan717829@foxmail.com  
ORCID：https://orcid.org/0009-0005-8411-7218

源代码采用 [MIT License](LICENSE)。
