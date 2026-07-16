# 轨迹 SFT：从 OpenAI-style 数据到 masked next-token loss

> 一句话定义：轨迹 SFT 用一条已经完成的交互轨迹做 teacher forcing，让模型学习“在当前历史上下文下，下一个 assistant token 应该是什么”；system、user、工具定义和工具返回保留为条件，但通常不作为预测目标。

## 0. 先给结论

一条工具轨迹可以抽象为：

```text
系统规则 → 用户问题 → 思考/动作 → 工具返回 → 思考/动作 → … → 最终回答
```

训练时最关键的 mask 是：

```text
[system]        mask=0    只作为条件
[tools]         mask=0    只作为条件
[user]          mask=0    只作为条件
[assistant思考] mask=1    参与 loss
[assistant调用] mask=1    参与 loss
[tool返回]      mask=0    保留在上下文，但不参与 loss
[assistant回答] mask=1    参与 loss
[padding]       mask=0    不参与注意力，也不参与 loss
```

因此，本项目训练的是：

\[
P_\theta(\text{assistant 思考、工具调用、最终回答}\mid
\text{系统、用户、工具定义、此前工具返回})
\]

## 1. 输入数据：OpenAI-style JSONL

### 1.1 文件级约定

- 文件使用 JSONL：一行是一条完整轨迹，各行相互独立。
- 每条记录至少包含 `messages`；需要工具时同时包含 `tools`。
- 工具由 JSON Schema 描述；assistant 通过 `tool_calls` 发出调用，工具结果通过 `role=tool` 和相同的 `tool_call_id` 返回。
- OpenAI 官方 SFT 文档也采用“一行一个 JSON 结构”的 JSONL 和 Chat Completions 消息格式，并给出了 `tools`、`assistant.tool_calls` 的训练样例。[OpenAI SFT 数据格式](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)
- `reasoning_content` 是本项目源数据使用的扩展字段，不是 OpenAI Chat Completions 基础消息格式的必需字段。它用于显式保存要监督的中间推理；若不训练显式推理，可以省略。

### 1.2 一条完整轨迹

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "query_order",
        "description": "查询订单状态",
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": {"type": "string"}
          },
          "required": ["order_id"],
          "additionalProperties": false
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "你是物流分析助手，必须基于工具结果回答。"},
    {"role": "user", "content": "订单 A102 当前是什么状态？"},
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "需要先查询订单状态。",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "query_order",
            "arguments": "{\"order_id\":\"A102\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"order_id\":\"A102\",\"status\":\"运输中\"}"
    },
    {
      "role": "assistant",
      "reasoning_content": "工具结果显示状态为运输中。",
      "content": "订单 A102 当前处于运输中。"
    }
  ]
}
```

OpenAI 的函数调用流程也是“提供工具定义 → 模型返回 tool call → 应用执行工具 → 将带关联 ID 的工具输出追加到上下文 → 模型继续回答”。[OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)

### 1.3 本项目真实样本：从 2,028 条候选追溯到 OpenAI 原始记录

`train_candidates_16k_2028.jsonl` 本身不是原始 OpenAI 格式，而是已经转换好的 ms-swift agent JSONL。它由 27 条 SQL 强验证轨迹和 2,001 条待复核轨迹组成。为了让本材料保持框架无关，这里从其中选择一条强验证 train 样本，再追溯到同一条原始 OpenAI-style 记录。

样本定位与质量信息：

| 字段 | 值 |
|---|---|
| source | `qwen3.6-27B_20260628_v41_openai.jsonl:199` |
| task_id | `task_000201` |
| split | `train` |
| 类型 | DWH/SQL 聚合 |
| 质量证据 | `sql_result_verified` |
| Qwen3.5 输入长度 | 3,464 tokens |
| 监督长度 | 416 tokens |
| 完整工具调用 | 4 次 |

真实轨迹的问题是：

```text
我想查一下 2026-06-19 那天运单轨迹事件里所有停留时长（分钟）总共是多少。
```

下面是同一条原始 OpenAI 轨迹的关键步骤节选。为便于汇报，只省略了与数学过程无关的目录枚举和数据表列表；表结构、聚合 SQL、工具结果和最终答案保持原义。

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个物流数据分析师。你可以使用 Bash 工具执行 sqlite3 命令查询数据库。分析数据时，先思考需要什么数据，再写 SQL 查询，最后根据查询结果给出分析结论。"
    },
    {
      "role": "user",
      "content": "我想查一下2026-06-19那天运单轨迹事件里所有停留时长（分钟）总共是多少。"
    },
    {
      "role": "assistant",
      "content": "\n\n",
      "reasoning_content": "fact_waybill_event 表可能是运单轨迹事件表，需要先检查表结构。",
      "tool_calls": [
        {
          "id": "call_schema",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\":\"sqlite3 /workspace/logistics.sqlite '.schema fact_waybill_event'\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_schema",
      "content": "CREATE TABLE fact_waybill_event (..., event_time TEXT, ..., dwell_minutes INTEGER);"
    },
    {
      "role": "assistant",
      "content": "\n\n",
      "reasoning_content": "表中存在 dwell_minutes 字段，现在按 event_time 的日期过滤并求和。",
      "tool_calls": [
        {
          "id": "call_sum",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\":\"sqlite3 /workspace/logistics.sqlite \\\"SELECT SUM(dwell_minutes) FROM fact_waybill_event WHERE date(event_time) = '2026-06-19';\\\"\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_sum",
      "content": "6639\n"
    },
    {
      "role": "assistant",
      "reasoning_content": "查询结果表明总停留时长为 6639 分钟。",
      "content": "2026-06-19 当天运单轨迹事件中所有停留时长的总和为 **6639 分钟**。"
    }
  ]
}
```

这条真实轨迹的监督区域为：

| 步骤 | 角色 | 是否参与 loss | 解释 |
|---|---|:---:|---|
| 系统规则 | system | 否 | 训练条件 |
| 用户问题 | user | 否 | 训练条件 |
| “检查表结构”推理 + schema tool call | assistant | 是 | 学习分析动作 |
| `CREATE TABLE ... dwell_minutes` | tool | 否 | observation，保留为上下文 |
| “按日期求和”推理 + `SELECT SUM(...)` | assistant | 是 | 学习 SQL 决策 |
| `6639` | tool | 否 | observation，保留为上下文 |
| “6639 分钟”最终回答 | assistant | 是 | 学习基于结果作答 |

因果关系非常明确：模型生成 `SELECT SUM(...)` 时还看不到未来的 `6639`；生成最终答案时，`6639` 已经作为过去的 observation 出现在上下文中。

### 1.4 原始 OpenAI 与 2,028 条训练文件的对应关系

| 原始 OpenAI-style 字段 | 2,028 条文件中的训练表示 |
|---|---|
| `assistant.reasoning_content` | 合并进 assistant 的 `<think>…</think>` |
| `assistant.tool_calls[]` | 独立的 `role=tool_call` 消息 |
| `role=tool` | `role=tool_response` |
| `assistant.content` | assistant 内容，最终答案继续参与 loss |
| 原始记录缺少顶层 `tools` | 转换时补齐 bash/read/write/edit 工具 schema |

因此：

- 若讲输入数据协议，应展示上面的原始 OpenAI-style 记录；
- 若讲本项目送入 ms-swift 的实际文件，再展示转换后的 agent 消息；
- 两者描述的是同一条轨迹，只是外部协议与框架内部表示不同。

### 1.5 数据不变量

一条轨迹进入训练前至少满足：

1. `messages` 按真实发生顺序排列。
2. 每个 `assistant.tool_calls[j].id` 唯一。
3. 每个工具返回的 `tool_call_id` 引用先前存在且尚未响应的调用。
4. 工具名称和参数符合 `tools` 中的 JSON Schema。
5. 工具调用与工具返回闭环，没有悬空调用或重复返回。
6. 轨迹以可监督的 assistant 输出结束，通常是最终回答。
7. validation/test 的同题变体不进入 train。

## 2. 轨迹 SFT 与普通 SFT

普通问答 SFT 通常是：

```text
user → final answer
```

轨迹 SFT 则是：

```text
user → assistant action₁ → observation₁ → assistant action₂
     → observation₂ → … → final answer
```

两者优化目标仍是 next-token cross-entropy。区别不在“换了一种 loss”，而在于：

- 输入序列包含完整交互历史；
- 监督区域包含中间 assistant 决策和工具调用；
- observation 作为环境给出的条件保留，但被 loss mask 掉；
- 模型不仅学习“答什么”，还学习“何时调用什么工具、参数怎么写、看到结果后如何继续”。

轨迹 SFT 不是在线强化学习：训练阶段通常不重新执行工具，也不让当前模型自由探索；它只是重放已记录的轨迹。

## 3. 数学化定义

### 3.1 序列化与分词

记一条 JSON 轨迹为 \(D\)。chat template/serializer 将角色、工具 schema、消息边界和正文序列化为字符串：

\[
s=S(D)
\]

tokenizer 再将其转为 token 序列：

\[
x_{1:T}=\tau(s),\qquad x_t\in\{1,\ldots,|V|\}
\]

其中 \(|V|\) 是词表大小，\(T\) 是序列总长度。角色标记、工具 JSON、换行和标点都会占 token，不能只按字符数估算训练长度。

### 3.2 因果注意力

decoder-only Transformer 使用因果注意力。位置 \(t\) 只能读取当前位置及其之前的 token：

\[
A_{t,s}=\begin{cases}
0,&s\le t\\
-\infty,&s>t
\end{cases}
\]

所以：

- 生成工具调用时，模型看不到未来的工具返回，不存在未来信息泄漏；
- 生成工具返回之后的推理和答案时，过去的工具返回是可见上下文；
- observation 被 loss mask 并不等于从输入中删除。

Transformer 的注意力基础见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

### 3.3 loss mask

为每个 token 定义监督指示量：

\[
m_t=\begin{cases}
1,&x_t\text{ 属于要学习的 assistant 输出区域}\\
0,&x_t\text{ 属于 system/user/tools/observation/padding}
\end{cases}
\]

训练实现通常构造 labels：

\[
y_t=\begin{cases}
x_t,&m_t=1\\
\text{IGNORE\_INDEX},&m_t=0
\end{cases}
\]

在 PyTorch/Hugging Face 体系中，`IGNORE_INDEX` 常取 `-100`。它只控制交叉熵是否计分，不控制 token 是否进入上下文。

下表采用本项目当前的监督策略。其他项目如果不希望显式训练推理过程，可以把 assistant 推理区域也设为 \(m_t=0\)，但工具调用和最终回答是否监督仍要单独定义。

| 语义区域 | 输入中保留 | 参与 loss | 原因 |
|---|:---:|:---:|---|
| system 指令 | 是 | 否 | 是条件，不要求模型复述 |
| tools schema | 是 | 否 | 是可用动作空间 |
| user 问题 | 是 | 否 | 是条件，不训练模型模仿用户 |
| assistant 推理 | 是 | 是 | 学习中间决策过程 |
| assistant tool call | 是 | 是 | 学习工具选择和参数生成 |
| tool observation | 是 | 否 | 环境给定事实，不应由模型伪造 |
| assistant 最终回答 | 是 | 是 | 学习最终任务输出 |
| padding | 是 | 否 | 仅用于批处理对齐 |

特殊的角色起止 token 是否参与 loss 由 chat template 决定；上表描述的是语义区域，不绑定某个框架的特殊 token 细节。

### 3.4 下一个 token 的概率

模型在位置 \(t\) 输出对词表中每个 token 的 logits：

\[
z_t=f_\theta(x_{\le t})\in\mathbb{R}^{|V|}
\]

用 softmax 转为“下一个 token 是 \(v\)”的概率：

\[
p_\theta(x_{t+1}=v\mid x_{\le t})
=\frac{\exp(z_{t,v})}{\sum_{u=1}^{|V|}\exp(z_{t,u})}
\]

若一个极简词表的 logits 为 \([2,1,0]\)，正确 token 是第一个，则：

\[
p=\frac{e^2}{e^2+e^1+e^0}\approx0.665,
\qquad -\log p\approx0.408
\]

如果该位置属于 observation，\(m_{t+1}=0\)，这项 loss 乘零；但 observation 仍会影响后续 assistant token 的条件概率。

### 3.5 单条轨迹的目标函数

teacher forcing 使用数据中的真实历史，而不是模型自己采样的历史。单条轨迹的 masked negative log-likelihood 为：

\[
\mathcal{L}(D;\theta)
=-\frac{1}{N_D}
\sum_{t=1}^{T-1}m_{t+1}
\log p_\theta(x_{t+1}\mid x_{\le t})
\]

其中：

\[
N_D=\sum_{t=2}^{T}m_t
\]

是实际参与监督的 token 数。等价地，训练最大化所有监督 token 的条件似然：

\[
\prod_{t:m_t=1}p_\theta(x_t\mid x_{<t})
\]

### 3.6 批次 loss

对 batch \(\mathcal{B}\) 中的轨迹，常用 token-level mean：

\[
\mathcal{L}_{\mathcal{B}}
=-\frac{
\sum_{i\in\mathcal{B}}\sum_t m_{i,t}
\log p_\theta(x_{i,t}\mid x_{i,<t})
}{
\sum_{i\in\mathcal{B}}\sum_t m_{i,t}
}
\]

因此“轨迹条数相同”不代表训练权重相同：监督 token 多的轨迹通常贡献更多项。若框架采用先按样本平均再按 batch 平均，权重会不同，必须在实验配置中明确。

## 4. 从 JSON 到一次参数更新

### Step 1：结构校验

验证角色、tool call ID、工具 schema、参数类型、调用/返回闭环和最终回答。结构错误不能靠 mask 修复。

### Step 2：语义筛选

确认最终答案正确、工具结果支持结论、工具调用没有使用未来信息。`verdict=correct` 只是证据之一，不应替代可重放验证或人工复核。

### Step 3：序列化

用目标模型自己的 chat template 把 `tools + messages` 转成模型真实看到的序列。训练和推理必须使用同一套角色标记、工具调用表示和 system 注入规则。

### Step 4：tokenize 与截断

得到 `input_ids`。若超过最大长度，优先按完整“调用—返回—后续结论”片段裁剪或筛除；不能截断到只剩调用、没有 observation，或保留答案却删除支撑证据。

### Step 5：构造三种 mask

- causal mask：阻止读取未来 token；
- padding mask：阻止读取补齐 token；
- loss mask/labels：只让 assistant 推理、动作和最终回答计入 loss。

三者作用不同，不能混用。

### Step 6：teacher-forced forward

模型一次并行读取整条训练序列，在每个位置输出下一 token 的 logits。虽然计算并行，概率分解仍是自回归的。

### Step 7：masked cross-entropy

只对 labels 不等于 `IGNORE_INDEX` 的位置计算交叉熵并求平均。observation、用户问题等位置的直接梯度贡献为零。

### Step 8：反向传播与优化

计算：

\[
g=\nabla_\theta\mathcal{L}_{\mathcal{B}}
\]

再由 AdamW 等优化器更新参数。梯度累积只是合并多个 micro-batch 的梯度后再执行 optimizer step，不改变目标函数本质。

### Step 9：LoRA 更新

采用 LoRA 时，基础权重 \(W_0\) 冻结，只训练低秩增量：

\[
W=W_0+\frac{\alpha}{r}BA
\]

其中 \(A,B\) 可训练、秩为 \(r\)。loss 与全量微调相同，区别只是可训练参数集合不同。参见 [LoRA](https://arxiv.org/abs/2106.09685)。

## 5. observation 到底发生了什么

假设轨迹是：

```text
用户问题 U → assistant 工具调用 A₁ → 工具返回 O₁ → assistant 回答 A₂
```

训练目标包含：

\[
\mathcal{L}
=-\log P_\theta(A_1\mid U)
-\log P_\theta(A_2\mid U,A_1,O_1)
\]

但不包含：

\[
-\log P_\theta(O_1\mid U,A_1)
\]

因为 \(O_1\) 是环境输出，不是 assistant 应生成的动作。

所以“mask 掉 observation”的准确含义是：

> observation 的 token 仍在 `input_ids` 中，后续 token 可以注意到它；只是它在 `labels` 中被置为 `IGNORE_INDEX`，不产生直接 loss。

如果删除 observation，模型无法学习“根据工具结果继续推理”；如果让 observation 参与 loss，模型会浪费容量模仿数据库、文件或终端输出，并可能学会伪造环境结果。

## 6. 训练与推理的差异

### 训练：离线重放

```text
整条轨迹已知 → teacher forcing → 并行计算 assistant 目标 loss → 不实际执行工具
```

### 推理：在线闭环

```text
用户问题 → 模型逐 token 生成工具调用 → 外部执行器运行工具
→ 工具结果追加到上下文 → 模型继续生成 → 最终回答
```

这会产生 exposure gap：训练时模型总能看到正确历史，推理时却可能先生成错误工具调用，使后续上下文偏离训练分布。因此必须评测工具选择、参数、错误恢复和最终答案，不能只看训练 loss。

## 7. 数据准备的完整闭环

### 7.1 格式层

- JSON 可解析、角色合法；
- tool call/response 闭环；
- 工具名、参数键、类型和必填项符合 schema；
- 存在最终 assistant 输出；
- 序列化后存在监督 token。

### 7.2 正确性层

- DWH/SQL：重放 agent SQL，与 verification SQL 或 gold 结果比较；
- KB：指定文档被访问，答案由对应段落支持；
- 报告：输入数据、计算过程和结论一致；
- 工具报错：区分成功恢复与最终失败；
- 不把“回答很长”“生成了文件”当作正确性证明。

### 7.3 去重与防泄漏

先按 task_id 固定切分 train/validation/test，再选择同题不同版本。否则同一问题的不同轨迹可能同时出现在训练集和测试集，评测结果会虚高。

### 7.4 长度与算力

训练成本由 token 数而不是文件条数决定。工具返回经常占大部分上下文，却不参与 loss，所以应同时统计：

\[
\text{监督密度}=\frac{\text{supervised tokens}}{\text{input tokens}}
\]

监督密度过低意味着大量显存和计算用于读取 observation，只有少量 token 提供梯度。

### 7.5 安全与可移植性

- 去除密钥、令牌和个人信息；
- 隔离网络命令与破坏性命令；
- 将机器相关绝对路径做一致映射；
- 工具 schema、prompt、tool call 和 tool response 中的路径同步替换。

## 8. 评测指标

| 层级 | 指标示例 |
|---|---|
| 格式 | JSON/tool call 可解析率、schema 合法率、调用返回闭环率 |
| 动作 | 工具选择准确率、参数准确率、执行成功率、平均调用步数 |
| 结果 | 数值/表格 exact match、文档证据支持率、最终答案正确率 |
| 系统 | 端到端任务成功率、时延、token/工具成本、错误恢复率 |

评测必须使用按 task_id 隔离的 held-out 数据，并保留不经过工具的普通能力回归集，检查灾难性遗忘。

## 9. 本项目的实际映射

当前 Qwen3.5-9B 轨迹 SFT 实现采用：

- 原始层：OpenAI-style `messages + assistant.tool_calls + role=tool`；
- 推理扩展：`assistant.reasoning_content`；
- 训练层：推理序列化为 `<think>…</think>`，工具调用转为 agent template 的 tool-call 表示；
- loss：assistant 推理、tool call 和最终回答参与，tool response 不参与；
- 工具 schema：bash/read/write/edit 注入 system 上下文；
- 微调：LoRA 或全参数训练，目标函数都是 masked next-token cross-entropy。

上述行为已由 [`verify_trajectory_encoding.py`](../scripts/verify_trajectory_encoding.py) 检查，并在真实 Qwen3.5 模板上确认：tool call 出现在 supervised labels 中，tool response 只在 input 中。OpenAI 到训练消息的转换见 [`prepare_trajectory_sft.py`](../scripts/prepare_trajectory_sft.py)。

当前候选文件是“格式严格候选合集”，不等于所有答案均已证明正确；正式训练还要应用正确性筛选、task 级切分和长度上限。

## 10. 汇报时可直接使用的五句话

1. 轨迹 SFT 本质仍是自回归 next-token prediction，不是另一种神秘 loss。
2. 完整轨迹被序列化为一个因果 token 序列；模型每一步只能看过去，不能看未来工具结果。
3. system、用户问题、工具定义和 observation 都保留为上下文，但通过 loss mask 不计分。
4. assistant 的中间推理、工具调用和最终回答参与交叉熵，所以模型既学习答案，也学习行动过程。
5. 好的轨迹训练不仅要格式正确，还要结果可验证、任务不泄漏、长度可承受，并在在线工具闭环中评测。

## 11. 常见误解

| 误解 | 正确说法 |
|---|---|
| mask observation 就是删除 observation | 错。它仍在输入中，只是不计算该位置的 loss |
| 轨迹 SFT 会在训练时实时执行工具 | 通常不会；训练重放记录，推理才执行工具 |
| loss 只训练最终答案 | 本项目还训练中间推理和工具调用 |
| tool call 是普通字符串，格式不重要 | 错。它必须满足工具 schema，推理时才能执行 |
| 训练 loss 低就代表 agent 成功 | 错。还要测工具选择、参数、执行和最终任务成功率 |
| 格式合格就能直接全量训练 | 错。正确性、切分、长度和安全仍需筛选 |

## 参考资料

- [OpenAI：Supervised fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)
- [OpenAI：Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Vaswani et al.：Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hu et al.：LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
