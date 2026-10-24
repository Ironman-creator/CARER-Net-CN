# 幅度限制与零初始化组件消融

本目录对应中文版第4.7节与表7。主配置为原有Core，不按无tanh方案极小的测试均值差重新选模型。

## 一 直接复算数值

从仓库根目录运行：

```bash
python -m pip install -r requirements-statistics.txt
python component_ablation/analyze_published.py
```

该入口只读9组×200张图像的CSV，不读取照片或模型权重。输出为 `artifacts/component_statistics/` 中的均值、种子间样本标准差、4项配对区间及核对报告。使用单个NumPy随机流、seed=20261021、每128次一批的交叉采样、10,000次重采样，顺序为Core−无tanh的Dice/细血管召回，再Core−随机输出初始化的两个指标，精确对应原区间。

## 二 文件

- `records/`：30个CSV，包括9份训练日志、9份验证逐图结果、9份测试逐图结果和3份汇总；同时保留配置、初始等价诊断、划分、验证选点、数据与模型封存哈希。
- `engine/`：完成GTX 1650本机实验的冻结代码，包括全部15个基础依赖文件及历史对照记录。
- `engine_4070/`：原RTX 4070实验包的完整源码快照，科学日程相同，预检使用至少10 GiB名义显存门槛。
- 两个引擎各自的 `package_manifest.json` 核对其原始快照；整个新中文仓库另有根目录 `RELEASE_MANIFEST.json`。它们不是同一个发布清单。

`records` 中原机器路径是历史配置原文，不是新机器的数据路径。不要把原路径当作当前运行配置，也不要修改原记录以使校验通过。

## 三 模型与日程

三个方案均为32,779,990个总参数，其中适配器1,201个。`core`为灰度＋0.1×tanh(残差)，末层零初始化；`no_bound`只去掉tanh，保留0.1；`random_init`只保留末层卷积构造器的默认随机权重和偏置。

种子42/123/3407，512×512、batch=1、FP32，5轮模块预热＋25轮模块和末端联合更新。三方案在同一种子下使用同一GN权重、540/60划分及逐轮增强张量。按第0–30轮验证soft Dice选点，全部9个模型封存后统一测试200张图像。组件族为两个对比×两个指标，98.75%区间调整这4项比较，不与历史主实验区间混用。

零初始化使记录的插入检查图像输入与logit均与原主干相同。相对默认随机输出初始化，Core提高Dice 0.001110、细血管中心线召回0.002276，两个调整后区间为正。移除tanh的两个性能差异区间跨零。tanh约束增量，而非把最终输入截断到[0,1]。记录的无界训练批次最大增量1.83051不是测试图像统计。

## 四 重跑历史组件实验

统计复算不需要权重；严格重跑原实验必须另备以下三个历史GN检查点，程序会核对SHA-256：

| 文件 | SHA-256 |
|---|---|
| `FIVES_best_ResAttentionUNetGN_seed42.pth` | `b47245f4b3e1f0557aeb8b5f38db0e297e8268a31b6c076dab4f30e8aaca566c` |
| `FIVES_best_ResAttentionUNetGN_seed123.pth` | `45e44c7c842db1ce6f3ff49488e01cad97b7c01fc52be7d143d4639226c9ace7` |
| `FIVES_best_ResAttentionUNetGN_seed3407.pth` | `4d50de4405c51d34736e7b3475e5c12067848526f4543cba9fe4b201bcf67f08` |

它们未随Git源码分发，可联系通讯作者核对取得途径。主实验计划器可从头训练新的GN模型，但新检查点不是这三个固定来源，不应改名冒充。若在新主干上重复组件研究，应另存新实验协议和结果，不能替换历史表7。

以RTX 4070为例，保留 `engine_4070` 完整目录，将三个检查点放在某实验根目录的 `weight_files_fives_v2_screen/`，数据按主文档准备。在仓库根目录可执行：

```bash
python component_ablation/engine_4070/component_runner.py check-package
python component_ablation/engine_4070/component_runner.py preflight --project-root /path/to/experiment --data-root /path/to/data --output-root artifacts/components_new --workers 2 --device cuda
python component_ablation/engine_4070/component_runner.py train --project-root /path/to/experiment --data-root /path/to/data --output-root artifacts/components_new --workers 2 --device cuda --arm core --seed 42
```

对三个方案、三个种子完成全部9次train后执行seal，再逐一test：

```bash
python component_ablation/engine_4070/component_runner.py seal --output-root artifacts/components_new
python component_ablation/engine_4070/component_runner.py test --project-root /path/to/experiment --data-root /path/to/data --output-root artifacts/components_new --workers 2 --device cuda --arm core --seed 42
python component_ablation/engine_4070/component_summary.py --output-root artifacts/components_new --phase test
```

上面train/test各仅示例一个方案、一个种子，完整实验需遍历9组；seal必须位于全部训练之后、任一新测试之前。已有原始启动器及 `README_4070.md` 保留在快照内，使用它们时请先核对默认父目录和Python路径。

`engine/`本机版本把名义显存门槛换为已记录的512像素FP32容量与断点恢复实测，并及时释放已记录的批次张量；分辨率、损失、增强、划分和训练轮次不变。其容量记录绑定当时GPU与PyTorch环境，不能作为另一台机器已通过容量测试的证明。因此4070读者使用 `engine_4070`，不要手工修改本机报告中的GPU名称或passed字段。

原严格 `component_summary.py --phase test` 还会核验9个模型权重及封存记录，不是仅凭CSV运行的工具。没有权重时应使用本目录新增的 `analyze_published.py`，其报告明确区分数值复算与模型权重核验。
