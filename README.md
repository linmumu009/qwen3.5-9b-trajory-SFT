# qwen3.5-9b-trajory-SFT

面向 Qwen3.5-9B 的轨迹数据整理、SFT 数据构建、训练配置与评测工程仓库。项目将开源 benchmark 与沙箱轨迹分仓管理，并沉淀商业模型轨迹转换、判分、归档和服务器同步流程。

> 当前仓库首先建立进度记录和版本管理规范；训练代码、配置与可公开产物后续按轮次纳入。原始数据、模型权重、服务器信息和其他敏感内容不提交到公开仓库。

## 当前进展

截至 2026-07-16：

- `datasets/` 已按 `open_source/` 与 `sandboxes/` 两仓分立方案重组。
- 149 个扁平化沙箱已拉回本地，共约 3.3 GB。
- 5 份商业模型轨迹已转换为 SFT 数据，并完成归档与 verdict 结果同步。
- 22 处训练、评测、看板和数据脚本路径已迁移。
- 4 个轨迹判分、转换和归档流程脚本已固化。
- 本机与训练服务器的目标存储结构和关键路径已完成核验。
- 已从 `qwen3.6-27b-msswift-sft` 复制并启动 `qwen3.5-9b-msswift-trajory-sft` 训练容器，`/workspace/sft` 已切换至独立工作目录，Ascend 环境验证通过。
- 已建立服务器统一工作根目录 `/data3/llin/trajory_sft`，后续轨迹 SFT 工作统一在该目录下组织。
- 已以官方 `ms-swift v4.3.0` A3 / Python 3.11 / CANN 9.0.0 镜像为基座，构建 `qwen3.5-9b-msswift-trajory-sft:v0.2.0` 训练镜像；新增依赖仅使用 GitCode 与阿里云 PyPI 镜像。
- 已确认本机芯片为 Atlas 900 A3（`ascend910_9391`）；官方 A2 镜像不兼容本机动态算子内核，不用于后续训练。
- Qwen3.5 NPU 快速线性注意力链路已补齐；Qwen3.5-9B 真实权重前向、LoRA 反向传播及一次 AdamW 优化器更新均在 NPU 上验证通过。
- 已删除旧的同名容器并用 v0.2.0 镜像重建，容器名与 `/data3/llin/trajory_sft:/workspace/sft` 挂载保持不变。
- 已将 24 个未筛选 OpenAI 轨迹 JSONL 复制到项目数据目录，共 36,000 条、约 3.4 GB；原始副本、verdict、manifest 和处理产物分目录保存。
- 已实现 OpenAI `reasoning_content/tool_calls/tool` 到 ms-swift `assistant/tool_call/tool_response` 的轨迹转换、身份对齐、结构审计和候选目录统计。
- 8 条 `correct` 完整工具轨迹已通过 Qwen3.5 loss mask 验收和真实 ms-swift LoRA 训练步；工具调用参与 loss，tool response 不参与 loss。
- 36,000 条中有 15,072 条被规则 judge 标为 `correct`；经工具闭环、最终回答、工具参数键和值类型严格校验后得到 14,333 条候选。
- 已为转换数据补齐 bash/read/write/edit 顶层工具 schema，并确认 Qwen3.5 模板会把工具契约注入 system；tool call 参与 loss，tool response 保留上下文但不参与 loss。
- 已完成 14,333 条候选的真实 Qwen3.5 token 与监督密度审计：中位长度 25,415，5,521 条超过 32K 且占 69.6% 输入 token，加权监督占比仅 18.59%。
- 已生成 4,259 条元数据级复判规划池，按 `(version, task_id)` 去重、每题每阶段最多 3 条，并按 task_id 固定切分 train/validation/test；其中 4,179 条弱证据样本仍需复判。
- Qwen3.5-9B LoRA rank 16 单卡实测峰值：4K 23.45 GiB、8K 25.80 GiB、16K 34.73 GiB、24.58K 45.66 GiB、32K 59.95 GiB；24K 仅完成对照实验，当前生产长度上限恢复并保持为 16K。
- 容器内 PyTorch 可见 16 个 NPU 设备；全参数 SFT 具备进一步验证条件，但尚未完成 ZeRO-3/FSDP 显存和稳定性实测，不能用 LoRA 峰值代替全参数结论。
- 已完成 `core_8k` 893 条候选的首轮保守复判：68 条自动通过、2 条自动拒绝、823 条保留人工/独立模型复核；其中 117 条因同一 `(version, task_id)` 存在 manifest 冲突而禁止自动决策。
- 自动通过记录中只有 train 切分的 61 条可作为下一步训练种子候选；validation 5 条和 test 2 条继续隔离，当前尚未物化为正式训练集。
- 已将全部 14,333 条严格格式候选物化为单一 ms-swift agent JSONL（约 1.2 GB），逐条补齐顶层工具 schema 并保持完整 `assistant/tool_call/tool_response` 轨迹；该文件是候选合集，不等同于最终训练集。
- 已将上游 35,999 条评测索引中的 15,085 条 `verdict=correct` 与 36,000 条原始数据精确关联，抽取为单一完整 OpenAI 轨迹 JSONL（约 1.3 GB）；上游缺少 verdict 的 1 条原始记录未纳入。
- 已完成上游 15,085 条 correct 的第一轮确定性筛选：14,333 条通过严格轨迹结构门槛，4,259 条进入去重、防泄漏和长度规划池；已物理隔离 27 条 SQL 强验证 train、2,001 条 16K 内 train 复核池、250 条 held-out 与 1,981 条 16–32K 长轨迹。
- Qwen3.5 经验换算中位数约为 1 token:1.91 字符、全量加权约 1:2.03；本轮 14,333 条中 14,332 条直接复用既有真实 token，仅 1 条使用校准估算且未跨长度分层边界。
- 已将 27 条强验证 train 与 2,001 条待复判 train 原样合并为单一 `train_candidates_16k_2028.jsonl`；held-out 和长轨迹未混入，该文件是候选合集而非复判完成的 `train_ready` 数据。
- 已将 27 条 SQL 结果强验证轨迹另存为 `train_smoke_strong_verified_16k_27.jsonl`，用于优先验证 ms-swift 训练链路；该文件不等同于零瑕疵正式训练集。
- Transformers/FSDP2 全参链路已完成故障定位：无 CPU offload 时可运行 8 个优化器步但在长样本反向阶段 OOM；仅激活卸载在第 2 步 OOM；参数与激活全卸载会因 HCCL 不支持 CPU DTensor collective 而在梯度裁剪阶段退出，因此不再作为本轮生产方案。
- 已切换至镜像内置的 Megatron-Core 0.16.0 + MindSpeed 0.16.0，采用 `TP4 / PP2 / DP2`、sequence parallel、distributed optimizer 和 full recompute；Qwen3.5-9B 的 32 层、16 个注意力头、4 个 KV 头均与该拓扑整除匹配。
- 已从 2,028 条候选中按真实 Qwen3.5 token 选择最长 32 条（15,913–16,374 tokens）完成两步全参压力测试：两步均成功，框架显存最高 27.85 GiB，`npu-smi` 逐卡物理峰值最高 34.32 GiB，退出码为 0。
- 16 卡 Qwen3.5-9B 全参轨迹 SFT 正式任务已正常完成 150/150 步，退出码为 0，总训练时长 1 小时 16 分 38 秒；10 份 MCore 检查点均完整保存，步数为 15、30、45、60、75、90、105、120、135、150，每份约 172 GiB。
- 已将 10 份 MCore 检查点全部导出为独立 Qwen3.5 Hugging Face BF16 safetensors 模型，统一位于 `/data3/llin/trajory_sft/exports/qwen35_9b_megatron_16k_2028_150steps_hf`；每份含 4 个权重分片、760 个索引张量，权重文件共 18,820,260,968 字节，不含 optimizer、RNG 或 `.distcp`。
- 10 份导出模型已全部通过 safetensors 头部、索引、配置、tokenizer、chat template 和 processor 验收，10 个权重分片指纹均唯一；`checkpoint-150-hf` 已通过 `swift deploy` 的 OpenAI 兼容 `/v1/models` 与 `/v1/chat/completions` HTTP 200 实测，临时服务随后已关闭并释放 NPU。
- 已新增框架无关的轨迹 SFT 原理材料，从 OpenAI-style JSONL、因果注意力、loss mask、next-token softmax/交叉熵推导到参数更新、在线工具闭环与评测，明确 observation“保留为上下文但不参与 loss”。
- 已从 2,028 条 16K train 候选中选取一条 3,464-token SQL 强验证轨迹，追溯同源 OpenAI 原始记录并作为材料实例，逐消息标注 loss mask 和 OpenAI→ms-swift 映射。
- 已新增以 `task_000201` 为贯穿案例的离线单页 HTML 教学报告，按 9 步严格推导 OpenAI-style 轨迹到 Qwen3.5 masked cross-entropy 和参数更新，并明确 3,464 个输入 token 中仅 416 个直接参与 loss。
- 已修订 Markdown 版轨迹 SFT 说明：统一采用可渲染的 `$$...$$`/`$...$` 数学语法，明确 loss mask 读取 role、模板分段与 loss 规则后如何生成等长 labels，并将 `task_000201` 逐步带入结构校验、编码、mask、前向、loss 和参数更新。
- 已进一步补全 next-token 的逐位置对齐：明确 `logits[j-1]` 预测 `input_ids[j]`，是否计分由目标侧 `labels[j]` 决定；用 `task_000201` 十个目标区段逐项标出 assistant/tool-call 监督与 tool-response `-100`，消除“前一个 token 被 mask 会连带屏蔽下一个 assistant”的误解。

## 版本记录

| 版本 | 日期 | 摘要 | 状态 | 详细说明 |
|---|---|---|---|---|
| v0.7.5 | 2026-07-16 | 补全 next-token 的预测方向、target-side mask 和真实轨迹逐区段监督关系 | 已完成 | [查看报告](updates/v0.7.5_20260716_205105_next-token预测与目标侧mask说明.md) |
| v0.7.4 | 2026-07-16 | 修复 Markdown 公式显示，补全 labels 构造细节和真实轨迹逐步对照 | 已完成 | [查看报告](updates/v0.7.4_20260716_202200_公式渲染与真实轨迹逐步说明.md) |
| v0.7.3 | 2026-07-16 | 新增真实轨迹 SFT 的手术式逐步讲解 HTML，覆盖证据、转换、mask、概率、梯度与验收 | 已完成 | [查看报告](updates/v0.7.3_20260716_171354_真实轨迹SFT逐步讲解HTML.md) |
| v0.7.2 | 2026-07-16 | 用 2,028 条候选中的真实强验证轨迹补充 OpenAI 格式和 mask 实例 | 已完成 | [查看报告](updates/v0.7.2_20260716_165238_真实轨迹示例与格式映射.md) |
| v0.7.1 | 2026-07-16 | 新增严谨精炼的轨迹 SFT 输入格式、数学目标与端到端过程说明 | 已完成 | [查看报告](updates/v0.7.1_20260716_163146_轨迹SFT过程说明材料.md) |
| v0.7.0 | 2026-07-16 | 完成 150 步全参轨迹 SFT，将 10 个检查点导出为可直接部署的 HF BF16 模型并通过 API 验收 | 已完成 | [查看报告](updates/v0.7.0_20260716_103324_十个训练检查点HF模型导出与API验收.md) |
| v0.6.0 | 2026-07-15 | 切换 Megatron/MindSpeed TP4/PP2/DP2，完成 16K 最长样本压力测试并启动 150 步全参轨迹 SFT | 已完成 | [查看报告](updates/v0.6.0_20260715_173250_Megatron并行全参轨迹SFT.md) |
| v0.5.2 | 2026-07-15 | 独立保存 27 条 16K 强验证轨迹作为训练 smoke 数据 | 已完成 | [查看报告](updates/v0.5.2_20260715_153810_27条强验证训练Smoke文件.md) |
| v0.5.1 | 2026-07-15 | 将 2,028 条 16K 内 train 候选合并为单一 ms-swift JSONL | 已完成 | [查看报告](updates/v0.5.1_20260715_153138_16K训练候选单文件合并.md) |
| v0.5.0 | 2026-07-15 | 完成上游 correct 的确定性筛选与四类物化，验证 24K 后恢复 16K 生产上限 | 已完成 | [查看报告](updates/v0.5.0_20260715_144458_上游correct确定性筛选与分层物化.md) |
| v0.4.3 | 2026-07-15 | 将上游评测标记 correct 的 15,085 条完整轨迹抽取为单一 OpenAI JSONL | 已完成 | [查看报告](updates/v0.4.3_20260715_141540_上游评测correct轨迹单文件抽取.md) |
| v0.4.2 | 2026-07-15 | 将 14,333 条严格候选物化为单一可直接读取的 ms-swift JSONL | 已完成 | [查看报告](updates/v0.4.2_20260715_134040_严格候选单文件物化.md) |
| v0.4.1 | 2026-07-15 | 完成 core_8k 首轮证据复判、抽样校准与可审计判定产物 | 已完成 | [查看报告](updates/v0.4.1_20260715_132034_core_8k首轮保守复判.md) |
| v0.4.0 | 2026-07-15 | 完成轨迹 token/质量审计、数据分层规划与 4K–32K NPU 显存实测 | 已完成 | [查看报告](updates/v0.4.0_20260715_112134_轨迹数据审计与长度分层方案.md) |
| v0.3.0 | 2026-07-14 | 完成轨迹数据副本、格式适配、训练 smoke 与筛选目录 | 已完成 | [查看报告](updates/v0.3.0_20260714_181936_轨迹数据适配与SFT训练smoke.md) |
| v0.2.0 | 2026-07-14 | 构建 Qwen3.5 A3 轨迹 SFT 派生镜像并完成训练级验证 | 已完成 | [查看报告](updates/v0.2.0_20260714_171447_Qwen3.5轨迹SFT训练镜像构建.md) |
| v0.1.4 | 2026-07-14 | 切换至官方 A3 ms-swift v4.3.0 镜像并完成 NPU 验证 | 已完成 | [查看报告](updates/v0.1.4_20260714_162727_官方A3镜像切换与NPU验证.md) |
| v0.1.3 | 2026-07-14 | 将训练容器工作区重新挂载到统一项目目录 | 已完成 | [查看报告](updates/v0.1.3_20260714_144951_容器工作目录重新挂载.md) |
| v0.1.2 | 2026-07-14 | 初始化服务器统一轨迹 SFT 工作目录 | 已完成 | [查看报告](updates/v0.1.2_20260714_144230_服务器工作目录初始化.md) |
| v0.1.1 | 2026-07-14 | 复制并验证 Qwen3.5-9B 轨迹 SFT 训练容器 | 已完成 | [查看报告](updates/v0.1.1_20260714_143608_服务器容器复制.md) |
| v0.1.0 | 2026-07-13 | 数据集重组、服务器同步及商业模型数据推送 | 已完成 | [查看报告](updates/v0.1.0_20260713_165608_数据集重组与服务器同步.md) |

## 更新规范

从本版本开始，每轮更新必须同时完成两件事：

1. 修改本 README：更新“当前进展”，并在“版本记录”表格顶部新增一行摘要。
2. 在 `updates/` 新增一份独立详细报告，不覆盖或复用旧报告。

详细报告文件名统一为：

```text
updates/v<主版本>.<次版本>.<补丁版本>_YYYYMMDD_HHMMSS_<主题>.md
```

版本号遵循以下约定：

- 主版本：仓库结构、数据协议或训练流程发生不兼容调整。
- 次版本：新增数据集、训练阶段、评测能力或完整流程。
- 补丁版本：问题修复、配置调整、小范围数据补充或文档订正。

每份详细报告至少包含版本、日期、更新类型、状态、更新概述、具体改动、涉及文件或产物、验证结果、遗留问题；涉及高风险操作时还应记录备份或回滚方式。可复制 [`updates/TEMPLATE.md`](updates/TEMPLATE.md) 开始新一轮记录。

## Git 提交建议

提交信息使用清晰的类型前缀，例如：

```text
feat: add trajectory conversion pipeline
fix: correct sandbox dataset path
data: add model trajectory manifest
docs: add v0.2.0 update report
```

每次正式版本更新完成后，建议创建对应 Git tag，例如 `v0.2.0`。

## 数据与安全

- 原始数据集、模型权重、训练输出和归档文件体积较大，不直接提交 Git。
- SSH 登录记录、私钥、公钥路径、服务器地址和访问凭据不提交公开仓库。
- `reference/` 中的第三方源码仅供本地参考，不作为本仓库代码提交。
