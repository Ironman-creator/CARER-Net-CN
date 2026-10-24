# FIVES 颜色适配器组件消融实验 4070运行说明

本包只补充检验两个问题：tanh 幅度限制是否影响分割表现，以及末层零初始化是否影响训练与结果。已有颜色、绿色通道、局部对比和简单投影对照已整理到 `historical`，不重复训练。新包不覆盖原部署代码，也不重新训练灰度源主干。

## 一 放到哪里

把整个 `FIVES_CARER_Component_Ablation_4070_20261021` 文件夹放进4070电脑原来的部署根目录，保留子文件夹结构。例如：

```text
RetinalSegmentation_4060_Deploy/
  .venv/Scripts/python.exe
  data/
  weight_files_fives_v2_screen/
  FIVES_CARER_Component_Ablation_4070_20261021/
    component_runner.py
    component_models.py
    component_summary.py
    run_logged.py
    frozen_base/
    reference_splits/
    historical/
    protocol.json
    开始训练.cmd
```

不要把新包内的文件逐个覆盖到旧项目。启动脚本默认把新包的父目录当作旧部署根目录，使用那里已有的 `.venv/Scripts/python.exe`，并将父目录下的 `data` 作为数据根目录。新产物只写入新包自己的 `outputs`。

运行前确认：RTX 4070 可用，旧环境能正常导入 PyTorch、NumPy、Pillow、SciPy、Matplotlib和tqdm；原生彩色 FIVES 数据仍在旧位置；建议至少预留10GB磁盘空间，保存九个模型的最佳与逐轮恢复检查点及结果；系统电源设置不会在长时间运行时自动睡眠。不要同时双击两次启动文件。

## 二 三个方案究竟改了什么

令 `g` 为固定灰度输入，`r` 为相同24通道颜色细节网络输出的原始残差。三个方案都使用 RGB 与9×9绿色局部对比、灰度旁路、0.1系数和同一GN主干；都没有误差精修器。

| 方案 | 送入主干的输入 | 唯一变化 |
|---|---|---|
| `core` | `g + 0.1*tanh(r)` | 完整核心配置；末层权重和偏置为零 |
| `no_bound` | `g + 0.1*r` | 只去掉 tanh，不去掉0.1或旁路，末层仍为零 |
| `random_init` | `g + 0.1*tanh(r)` | 只保留末层卷积构造器原有的默认随机权重与偏置 |

同一种子构造时，三个方案的主干和颜色特征参数一致；随机初始化方案只在最后输出卷积不同，也不额外消耗随机数改变前层。总参数都为32,779,990，新增模块都为1,201。输入相加后不做截断，因此校正输入可能略超出零至一；`core` 限制的是增量。

本包没有去掉灰度旁路。直接删旁路又保留零输出层，会使起始输入全零，不能据此当作独立的旁路收益证据。

## 三 固定九次训练与统一测试

种子固定为42、123、3407，每个种子训练三个方案，共九次。每次都从该种子的历史灰度GN检查点出发，完整微调30轮：前5轮只训练新模块，后25轮以0.0001训练模块、以0.00001训练末级解码块与输出层，其余主干冻结。每次使用保存的540幅训练、60幅验证划分；按验证 soft Dice 从第0至30轮中选择检查点，同分保留较早轮。

全部采用512×512、批量1、FP32和固定增强。测试使用固定0.5阈值，不搜索阈值，不自动增加种子，不因为结果不好而换检查点或重训源主干。显存不足或配置不符会停止，不会自动降分辨率、启用AMP或替换训练方式。

九次训练全部成功后，程序先生成 `training_seal.json`，封存九个验证选定模型及相关文件哈希，再统一对200幅官方测试图像运行固定评价。即使某个方案没有正收益，也保留并汇总全部结果。

训练和测试文件分别记录内容SHA256与按512像素预处理的像素哈希。FIVES官方两个目录可能存在同名但不同内容的照片，因此程序按实际图像内容检查跨集合重复，不以文件名相同直接判为泄漏。

这200幅图像在此前研究中已经查看，本次属于既有 FIVES 上的后验组件消融，不是新的独立确认队列。训练封存防止的是本轮看完新测试结果再选模型，不会抹去历史测试使用情况。

## 四 历史源检查点不能替换

旧部署根目录必须有以下三个文件，哈希须与 `protocol.json` 完全一致：

| 种子 | 相对旧根目录的路径 | SHA256 |
|---|---|---|
| 42 | `weight_files_fives_v2_screen/FIVES_best_ResAttentionUNetGN_seed42.pth` | `b47245f4b3e1f0557aeb8b5f38db0e297e8268a31b6c076dab4f30e8aaca566c` |
| 123 | `weight_files_fives_v2_screen/FIVES_best_ResAttentionUNetGN_seed123.pth` | `45e44c7c842db1ce6f3ff49488e01cad97b7c01fc52be7d143d4639226c9ace7` |
| 3407 | `weight_files_fives_v2_screen/FIVES_best_ResAttentionUNetGN_seed3407.pth` | `4d50de4405c51d34736e7b3475e5c12067848526f4543cba9fe4b201bcf67f08` |

预检同时确认元数据为 FIVES、`res_attention_unet_gn`、灰度512输入，并核对数据和历史划分。缺文件或哈希不符时，先保存报错并联系整理者；不要把其他权重改名覆盖，也不要重训一个新主干冒充这三个来源。

## 五 最简单的启动方法

在4070电脑双击 `开始训练.cmd`。默认顺序为：

```text
检查包完整性
预检训练数据 源检查点 与512像素GPU前向
依次完成三个种子乘三个方案的九次训练
封存全部九个验证选定检查点
生成验证汇总
依次运行九个固定测试
生成测试配对统计
自动生成不含大权重的回传ZIP
```

若某次训练失败，会记录失败并继续执行其他训练；全部训练走完后，只要还有失败，就不会封存，也不会开始正式测试。不要因个别结果不理想中途改协议。

日志实时显示在窗口，同时写入 `outputs/logs/pipeline_日期时间_进程号.log`。`pipeline_status.txt` 是最近一条状态，完整原因要看日志。脚本退出时恢复它临时设置的UTF-8与CUDA确定性环境变量。

## 六 分阶段运行与中断恢复

在新包目录打开PowerShell，可分别执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fives_component_ablation_4070.ps1 -Stage Preflight
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fives_component_ablation_4070.ps1 -Stage Train
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fives_component_ablation_4070.ps1 -Stage Test
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fives_component_ablation_4070.ps1 -Stage Summary
```

`Preflight` 只预检，不训练或测试。`Train` 预检并完成九次训练，成功后封存、生成验证汇总，不运行官方测试。`Test` 要求已存在完整封存，程序逐次核验后测试。`Summary` 汇总验证与测试，不训练；若只有训练结果，增加 `-SummaryPhase Validation`。

训练每轮保存恢复状态；断电或中断后，使用相同目录、代码、环境和设置，重新运行原来的命令。程序会从最后完整保存的轮次继续，必要时重做未完成的一轮；已经完成且校验通过的运行直接跳过，不重复训练。不要手动改日志、检查点或 `training_seal.json`。

如果旧根目录或数据根目录不同，可显式指定，路径有空格时保留引号：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fives_component_ablation_4070.ps1 -Stage All -ProjectRoot "D:\RetinalSegmentation_4060_Deploy" -DataRoot "D:\RetinalSegmentation_4060_Deploy\data"
```

默认 `-Workers 2 -Device cuda`。已有运行恢复时保持这些设置不变；不要把CPU运行和4070结果混在同一目录。脚本没有提供修改轮数、输入大小、损失权重或自动混合精度的开关。

## 七 跑完带哪些文件回来

最方便的方法：把新包目录下自动生成的 `FIVES_Component_Ablation_Return_日期时间.zip` 带回来，放进本机“实验数据”文件夹即可。它包含九次运行的训练日志、逐图结果、预测图、配对统计、数据和封存记录，以及回传文件哈希清单；大权重保留在4070电脑。

回传包生成前，会在4070端重新核验封存与全部最佳权重。接收电脑可以检查回传文件与阅读结果，但不含权重的ZIP不能独立重做模型权重校验；请保留4070端完整原件。若要在接收电脑再次完整核验，拷回完整 `outputs` 文件夹，或补拷九个 `best.pth`。目录包括：

```text
outputs/
  runs/                 九次训练配置 日志与验证逐图结果
  results/              九次官方测试的逐图指标与汇总
  summaries/validation/ 验证汇总及完整性状态
  summaries/test/       测试汇总 配对统计与审计报告
  logs/                 实时控制台日志与流水线状态
  checkpoints/          各方案各种子的best.pth与last.pth
  training_seal.json
  preflight.json
  training_data_manifest.json
  official_test_data_manifest.json
```

若手动拷贝，优先拷回 `runs`、`results`、`summaries`、`logs`、四个根目录JSON，以及 `checkpoints` 内全部九个 `best.pth`，保持原有相对路径。完整复核的汇总程序需要这些最佳模型来验证封存哈希；不能只拿表格回来就声称已经在接收电脑重新核验了模型。

`last.pth` 主要用于逐轮续跑，可以在确认全部完成后留在4070电脑。若九个 `best.pth` 也留在那里，仍可阅读已生成的CSV和报告，但接收电脑无法独立重跑当前完整性校验型测试汇总；需保留4070端完整原件，之后补拷最佳模型。不提供绕过封存校验的开关，也不要将缺权重的离线阅读标成模型已验证。

还建议把本包原有 `historical` 和 `protocol.json` 随同保留，以便把既有颜色证据与本次组件实验分开整理。

如果中途失败需要先返回诊断材料，可在新包目录运行下面的命令。生成的ZIP会明确标记不完整，不会当作完成实验：

```powershell
..\.venv\Scripts\python.exe .\make_return_bundle.py --allow-incomplete
```

## 八 已有局部对比和简单投影证据在哪里

`historical/历史局部对比结果.md` 汇总了全部七种既有方案：轮数匹配灰度、绿色通道、灰度CLAHE、1×1 RGB投影、仅RGB残差、仅局部对比残差、完整Core；对应原始表、逐图记录及来源哈希也一并保留。

简单1×1 RGB投影已有约0.898092 Dice，与Core约0.898350很接近，Core减投影的区域Dice区间跨零。不能把它省略后写成完整结构已明显优于简单颜色输入，也不能把区间跨零解释成已证明等效。

已有仅RGB残差为985个新增参数，而完整Core为1,201；两者不属于严格等容量的线索单因素消融。历史结构补充显示细血管中心线覆盖有一定差异，但其他结构终点并非都表现为明确改善。请保留完整历史材料，并用本轮固定实验回答幅度限制和初始化问题，不根据哪张表好看选择性汇报。
