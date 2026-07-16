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

$$
P_\theta(\text{assistant 思考、工具调用、最终回答}\mid
\text{系统、用户、工具定义、此前工具返回})
$$

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

配套的单页 HTML 教学报告见 [`trajectory_sft_task_000201_walkthrough.html`](trajectory_sft_task_000201_walkthrough.html)：它以本节同一条轨迹为例，逐步推导格式转换、Qwen 模板编码、loss mask、因果可见性、交叉熵和参数更新。

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

记一条 JSON 轨迹为 $D$。chat template/serializer 将角色、工具 schema、消息边界和正文序列化为字符串：

$$
s=S(D)
$$

tokenizer 再将其转为 token 序列：

$$
x_{1:T}=\tau(s),\qquad x_t\in\{1,\ldots,|V|\}
$$

其中 $|V|$ 是词表大小，$T$ 是序列总长度。角色标记、工具 JSON、换行和标点都会占 token，不能只按字符数估算训练长度。

### 3.2 因果注意力

decoder-only Transformer 使用因果注意力。位置 $t$ 只能读取当前位置及其之前的 token：

$$
A_{t,s}=\begin{cases}
0,&s\le t\\
-\infty,&s>t
\end{cases}
$$

所以：

- 生成工具调用时，模型看不到未来的工具返回，不存在未来信息泄漏；
- 生成工具返回之后的推理和答案时，过去的工具返回是可见上下文；
- observation 被 loss mask 并不等于从输入中删除。

Transformer 的注意力基础见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

### 3.3 loss mask / labels：它读取什么，输出什么，究竟 mask 什么

#### 3.3.1 它读取的不是一段纯文本，而是“带角色边界的训练记录”

本项目不是先把整条轨迹拼成纯文本，再靠关键词猜哪些地方要训练。实际入口是转换后的 ms-swift agent 记录：

```json
{
  "tools": "[bash/read/write/edit 的 JSON Schema]",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>", "loss": true},
    {"role": "tool_call", "content": "{...}", "loss": true},
    {"role": "tool_response", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\n\n最终答案", "loss": true}
  ]
}
```

然后由 Qwen3.5 对应的 ms-swift template 编码：

```python
processor = get_processor("/models/Qwen3.5-9B")
template = get_template(processor, loss_scale="default+ignore_empty_think")
template.set_mode("train")
encoded = template.encode(row)

input_ids = encoded["input_ids"]
labels = encoded["labels"]
```

构造 labels 时使用的信息是：

1. `messages[i].role`：区分 system、user、assistant、tool_call、tool_response；
2. chat template 产生的消息边界和特殊 token 区段；
3. 当前 `loss_scale="default+ignore_empty_think"` 的分段规则；
4. 转换记录中显式写入的 `loss` 属性。对不同 role，字段是否直接生效由模板实现决定，因此最终结果必须检查真实 labels，不能只看 JSON 字段名猜测；
5. batch collator 产生的 padding 区域。

它**不负责**判断 SQL 是否正确，也不会因为正文里出现 `6639` 就自动决定要不要监督。正确性由进入编码前的数据筛选解决；loss mask 只解决“哪一段是模型输出目标”。

#### 3.3.2 它输出两个等长数组

设真实 token 序列为 $x_{1:T}$。对每个 token 定义监督指示量：

$$
m_t=\begin{cases}
1,&x_t\text{ 位于要学习的 assistant/tool-call 区域}\\
0,&x_t\text{ 位于 tools/system/user/tool-response/padding 区域}
\end{cases}
$$

labels 的逐位置构造为：

$$
y_t=\begin{cases}
x_t,&m_t=1\\
-100,&m_t=0
\end{cases}
$$

因此通常有：

```text
len(input_ids) == len(labels)

input_ids[t] = 该位置真实 token id
labels[t]    = 同一个 token id，或者 -100
```

`-100` 是交叉熵的 `IGNORE_INDEX`：该位置不作为一道“预测题”计入 loss。它不删除 `input_ids[t]`，也不会阻止后续 token 注意到它。

| 语义区域 | `input_ids` 中是否存在 | `labels` 中是什么 | 直接参与 loss | 后续 assistant 能否读取 |
|---|:---:|---|:---:|:---:|
| tools schema | 是 | `-100` | 否 | 是 |
| system 指令 | 是 | `-100` | 否 | 是 |
| user 问题 | 是 | `-100` | 否 | 是 |
| assistant 推理 | 是 | 真实 token id | 是 | 是 |
| assistant tool call | 是 | 真实 token id | 是 | 是 |
| tool response / observation | 是 | `-100` | 否 | 是 |
| assistant 最终回答 | 是 | 真实 token id | 是 | 是 |
| batch padding | 是 | `-100` | 否 | 否，另由 attention mask 屏蔽 |

特殊角色起止 token 是否计入监督由 Qwen chat template 决定。上表描述语义区域；精确边界以 `template.encode(row)` 生成的真实 labels 为准。

#### 3.3.3 放到 `task_000201` 上逐段看

| 实际轨迹区段 | `input_ids` | `labels` | 训练含义 |
|---|:---:|---|---|
| 注入的 bash/read/write/edit schema | 保留 | `-100` | 告诉模型有哪些工具，不要求模型复述 schema |
| system：物流数据分析师规则 | 保留 | `-100` | 作为行为条件 |
| user：查询 2026-06-19 总停留分钟数 | 保留 | `-100` | 作为任务条件 |
| 前两轮目录/表枚举的 assistant 推理与调用 | 保留 | 真实 token id | 学习先定位数据库、再找表；公开节选省略了具体命令文本 |
| 前两轮目录/表枚举的工具返回 | 保留 | `-100` | 提供下一步依据，不训练模型模仿环境输出 |
| assistant：“检查 `fact_waybill_event` 表结构” | 保留 | 真实 token id | 学习为什么需要查 schema |
| tool call：`.schema fact_waybill_event` | 保留 | 真实 token id | 学习生成工具名、命令和参数 |
| tool response：含 `event_time`、`dwell_minutes` | 保留 | `-100` | 模型后续可据此写 SQL，但不预测数据库返回 |
| assistant：“按日期过滤并对 `dwell_minutes` 求和” | 保留 | 真实 token id | 学习从 schema 推到聚合动作 |
| tool call：`SELECT SUM(dwell_minutes) ...` | 保留 | 真实 token id | 学习正确 SQL |
| tool response：`6639` | 保留 | `-100` | 最终回答可以读取；该数值本身没有直接 target loss |
| assistant：“总和为 6639 分钟” | 保留 | 真实 token id | 学习忠实使用工具结果作答 |

用语义分区做一个缩微示意；方括号代表一段 token，不代表 Qwen tokenizer 的真实切词：

```text
input_ids:
[TOOLS] [SYSTEM] [USER] [THINK] [TOOL_CALL] [TOOL_RESPONSE:6639] [FINAL:6639分钟]

labels:
[ -100] [  -100] [-100] [真实id] [    真实id] [              -100] [       真实id]
```

真实编码结果为：

$$
T=3464,\qquad N_{supervised}=416,\qquad N_{masked}=3464-416=3048
$$

$$
\text{监督密度}=\frac{416}{3464}=12.0092\%\approx12.01\%
$$

验证代码不是凭文本估计，而是直接统计：

```python
supervised_ids = [label for label in labels if label != -100]
assert len(input_ids) == 3464
assert len(supervised_ids) == 416
```

它还把监督 token 解码回文本，确认 `<tool_call>` 位于监督区，而 `<tool_response>` 不在监督区。

#### 3.3.4 三种 mask 的职责必须分开

| 名称 | 读取什么 | mask 掉什么 | 是否删除正文 |
|---|---|---|:---:|
| causal mask | token 的先后位置 | 对当前位置屏蔽所有未来 token | 否 |
| padding/attention mask | batch 中的有效长度 | 屏蔽为了对齐而增加的 pad | 否 |
| loss mask / labels | role、模板分段、loss 规则 | 把非监督 target 设为 `-100` | 否 |

所以“mask 掉 observation”的完整说法是：**observation 在 labels 中被忽略，但在 input_ids 中保留；只要它位于过去，后续 assistant token 就能通过因果注意力读取它。**

### 3.4 下一个 token 的概率

模型在位置 $t$ 输出对词表中每个 token 的 logits：

$$
z_t=f_\theta(x_{\le t})\in\mathbb{R}^{|V|}
$$

用 softmax 转为“下一个 token 是 $v$”的概率：

$$
p_\theta(x_{t+1}=v\mid x_{\le t})
=\frac{\exp(z_{t,v})}{\sum_{u=1}^{|V|}\exp(z_{t,u})}
$$

若一个极简词表的 logits 为 $[2,1,0]$，正确 token 是第一个，则：

$$
p=\frac{e^2}{e^2+e^1+e^0}\approx0.665,
\qquad -\log p\approx0.408
$$

如果该位置属于 observation，$m_{t+1}=0$，这项 loss 乘零；但 observation 仍会影响后续 assistant token 的条件概率。

### 3.5 单条轨迹的目标函数

teacher forcing 使用数据中的真实历史，而不是模型自己采样的历史。单条轨迹的 masked negative log-likelihood 为：

$$
\mathcal{L}(D;\theta)
=-\frac{1}{N_D}
\sum_{t=1}^{T-1}m_{t+1}
\log p_\theta(x_{t+1}\mid x_{\le t})
$$

其中：

$$
N_D=\sum_{t=2}^{T}m_t
$$

是实际参与监督的 token 数。等价地，训练最大化所有监督 token 的条件似然：

$$
\prod_{t:m_t=1}p_\theta(x_t\mid x_{<t})
$$

### 3.6 批次 loss

对 batch $\mathcal{B}$ 中的轨迹，常用 token-level mean：

$$
\mathcal{L}_{\mathcal{B}}
=-\frac{
\sum_{i\in\mathcal{B}}\sum_t m_{i,t}
\log p_\theta(x_{i,t}\mid x_{i,<t})
}{
\sum_{i\in\mathcal{B}}\sum_t m_{i,t}
}
$$

因此“轨迹条数相同”不代表训练权重相同：监督 token 多的轨迹通常贡献更多项。若框架采用先按样本平均再按 batch 平均，权重会不同，必须在实验配置中明确。

## 4. 从 JSON 到一次参数更新

### Step 1：结构校验

- **一般操作**：验证角色、tool call ID、工具参数 JSON、调用/返回闭环和最终 assistant 输出。结构错误不能靠 mask 修复。
- **`task_000201` 实际发生的事**：读取 `qwen3.6-27B_20260628_v41_openai.jsonl:199`；确认共有 4 次 tool call，每次都有且只有一个引用正确 ID 的 tool response；Bash 参数可解析为 JSON object；轨迹以最终 assistant 回答结束。
- **本步产物**：一条结构闭合、可以继续做语义判断的 OpenAI-style 轨迹。此时还没有 token，也还没有 labels。

### Step 2：语义筛选

- **一般操作**：确认最终答案正确、工具结果支持结论、工具调用没有使用未来信息。`verdict=correct` 只是证据之一，不应替代可重放验证或人工复核。
- **`task_000201` 实际发生的事**：质量层为 `sql_result_verified`；schema observation 表明 `fact_waybill_event` 含 `event_time` 和 `dwell_minutes`；随后执行：

  ```sql
  SELECT SUM(dwell_minutes)
  FROM fact_waybill_event
  WHERE date(event_time) = '2026-06-19';
  ```

  工具返回 `6639`，最终回答也是 `6639 分钟`，问题日期、聚合字段、聚合函数、数值和单位一致。
- **本步产物**：不是“格式看起来像对”，而是一条关键结果有可验证证据的 train 样本。

### Step 3：序列化

- **一般操作**：先把 OpenAI-style 消息转成 ms-swift agent 消息，再用目标模型自己的 chat template 序列化。训练和推理必须使用同一套角色标记、工具调用表示和 system 注入规则。
- **`task_000201` 实际发生的事**：
  1. `assistant.reasoning_content` 被包进 `<think>...</think>`；
  2. 每个 `assistant.tool_calls[]` 被拆成独立 `role=tool_call`；
  3. `role=tool` 被改成 `role=tool_response`；
  4. 最终 assistant 内容继续保留；
  5. 顶层补入 bash/read/write/edit 工具 schema。

  其中关键聚合动作变为：

  ```json
  {
    "role": "tool_call",
    "content": "{\"name\":\"bash\",\"arguments\":{\"command\":\"sqlite3 ... SELECT SUM(dwell_minutes) ...\"}}",
    "loss": true
  }
  ```

- **本步产物**：`tools + messages` 仍是结构化对象；尚未变成 token。原始 call ID 已完成闭环校验，转换后的顺序消息不再依赖该 ID。

### Step 4：tokenize 与截断

- **一般操作**：Qwen chat template 先写入工具定义、角色标记、消息正文和边界标记，Qwen3.5 tokenizer 再生成 `input_ids`。若超过上限，优先整条筛除或按完整“调用—返回—后续结论”片段处理。
- **`task_000201` 实际发生的事**：真实 Qwen3.5 编码长度为 3,464 tokens，小于当前 16K 上限，因此整条保留，没有截断任何调用、observation 或最终答案。
- **本步产物**：长度为 3,464 的整数序列 `input_ids`。这 3,464 不等于原文字符数，其中包含工具 schema、角色特殊 token、JSON 标点和换行。

### Step 5：构造三种 mask

- **causal mask**：对这条轨迹，模型生成 `SELECT SUM(...)` 时可以看到此前的 schema observation，但不能看到未来工具返回 `6639`；生成最终答案时，`6639` 已经位于左侧历史，因此可见。
- **padding mask**：样本自身有效长度是 3,464；若所在 batch 的最大长度更长，collator 会补 pad，并让这些 pad 不进入注意力。补多少取决于当时 batch，不能从单条记录固定推断。
- **loss mask / labels**：template 依据 role 和分段规则生成等长 labels。tools/system/user/tool_response/pad 对应 `-100`；assistant 推理、tool call、最终回答对应真实 token id。

本样本的精确计数是：

```text
input_ids 总数                  3,464
labels != -100                   416  ← 直接计算 loss
labels == -100                 3,048  ← 只作条件或 padding
```

尤其要看清：工具返回里的 `6639` 在 `input_ids` 中存在，但它自己的 label 是 `-100`；最终回答中的“6639 分钟”属于 assistant，labels 保留真实 token id。

- **本步产物**：causal/attention 规则，以及长度与 `input_ids` 对齐的 labels。三者作用不同，不能混用。

### Step 6：teacher-forced forward

- **一般操作**：模型一次并行读取整条真实训练序列，在每个位置输出对“下一个 token”的 logits；虽然计算并行，因果 mask 仍保证概率分解是自回归的。
- **`task_000201` 实际发生的事**：

  | 正在预测的 assistant 区段 | 允许读取的真实前缀 | 不能读取的未来 |
  |---|---|---|
  | “需要检查表结构”及 schema 调用 | system、用户问题、此前探索 observation | schema 返回、`6639`、最终答案 |
  | `SELECT SUM(dwell_minutes)...` | 用户问题、此前动作、表结构中的字段 | 查询结果 `6639`、最终答案 |
  | “总和为 6639 分钟” | 前面全部轨迹，包括 tool response `6639` | 后续 token |

  teacher forcing 使用的是数据中的正确历史，而不是让当前模型先自由生成一遍 SQL。即使某个位置的 label 是 `-100`，模型通常仍会计算该位置 hidden state，使后续 supervised token 可以利用它。
- **本步产物**：每个序列位置上的词表 logits，形状概念上为 `[sequence_length, vocabulary_size]`。

### Step 7：masked cross-entropy

- **一般操作**：位置 $t$ 的 logits 预测位置 $t+1$ 的 token；实现对 logits 和 labels 做 shift 后，只保留 labels 不等于 `-100` 的位置。
- **`task_000201` 实际发生的事**：3,464 个位置中有 416 个 supervised labels。单样本在标准 non-ignored-token mean 口径下：

$$
\mathcal{L}_{000201}
=-\frac{1}{416}
\sum_{j\in A_{000201}}
\log P_\theta(x_j\mid x_{<j})
$$

$A_{000201}$ 包含 assistant 推理、工具调用 JSON/SQL 和最终答案对应的监督 token。tool response 中的 `6639` 不在这个集合；最终回答中的“6639 分钟”在这个集合。
- **本步产物**：该样本的 loss 分子与 416 个有效 target 计数。进入 batch 后，通常与其他样本的非忽略 token 一起归一化。

### Step 8：反向传播与优化

计算：

$$
g=\nabla_\theta\mathcal{L}_{\mathcal{B}}
$$

- **`task_000201` 实际发生的事**：如果它位于当前 micro-batch，它通过上述 416 个 target loss 对 batch 梯度作贡献。`6639` observation 没有自己的直接 loss，但会影响后续“6639 分钟”的 hidden state 和概率，因此会沿这条后续计算路径间接影响梯度。
- **不能误写成**“这一条轨迹单独更新一次参数”：实际 optimizer step 可能聚合多个样本、多个 data-parallel rank 和多个 gradient-accumulation micro-batch。
- **本步产物**：聚合、裁剪后的梯度，再由 AdamW 等优化器更新被允许训练的参数。梯度累积不改变目标函数本质。

### Step 9：确定更新全参数还是 LoRA 参数

- **全参数 SFT**：本项目正式 Megatron/MindSpeed 训练链路允许基础模型参数参与更新。
- **LoRA SFT**：本项目 smoke 链路也验证过 LoRA；此时基础权重 $W_0$ 冻结，只训练低秩增量：

$$
W=W_0+\frac{\alpha}{r}BA
$$

其中 $A,B$ 可训练、秩为 $r$。对 `task_000201` 而言，输入、labels、416 个监督 token 和 masked cross-entropy 都不变；变化的只是梯度最终允许更新哪些参数。参见 [LoRA](https://arxiv.org/abs/2106.09685)。

## 5. observation 到底发生了什么

假设轨迹是：

```text
用户问题 U → assistant 工具调用 A₁ → 工具返回 O₁ → assistant 回答 A₂
```

训练目标包含：

$$
\mathcal{L}
=-\log P_\theta(A_1\mid U)
-\log P_\theta(A_2\mid U,A_1,O_1)
$$

但不包含：

$$
-\log P_\theta(O_1\mid U,A_1)
$$

因为 $O_1$ 是环境输出，不是 assistant 应生成的动作。

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

$$
\text{监督密度}=\frac{\text{supervised tokens}}{\text{input tokens}}
$$

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
