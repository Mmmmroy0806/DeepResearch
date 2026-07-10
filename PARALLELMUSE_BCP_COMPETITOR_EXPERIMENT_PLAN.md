# ParallelMuse 竞品实验执行方案（BrowseComp-Plus + baseline retriever + gpt-oss-20b）

本文档是给执行 agent 的完整任务说明。执行 agent 不具备前序讨论上下文，因此本文档同时包含：背景与实验目的、已经完成的工作（代码已全部写好并验证）、已经确认和排除的设计决策、逐步执行协议、每步验收标准、以及所有已知的坑。

**一句话目标：在固定 baseline retriever 的前提下，用 ParallelMuse 方法（预算 8）+ gpt-oss-20b 跑 BrowseComp-Plus 前 100 条 query，产出可供 BrowseComp-Plus evaluator 评测的结果，作为 deepsearch_edi 论文的竞品条目。**

---

## 0. 背景：这个实验在论文里的位置

用户正在投稿 `deepsearch_edi` 论文（位于 `/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi`，**禁止修改该仓库任何文件**）。论文需要竞品对比实验，竞品之一是 ParallelMuse（Tongyi Lab 的推理时并行方法，arxiv 2510.24698）。

实验要回答的问题**不是**：

```text
ParallelMuse 自己联网搜索能做到什么？
```

而是：

```text
在使用与 deepsearch_edi 完全相同的 BrowseComp-Plus baseline retriever 的前提下，
ParallelMuse 这个竞品方法（搭载 gpt-oss-20b）的表现如何？
```

第一性原则：**固定 retriever，替换 agent/方法**。竞品可以决定搜什么 query、何时继续搜、如何聚合证据，但不能决定用什么搜索引擎、什么索引、top_k、是否联网、是否 rerank。

遇到不确定情况时的判断原则（按优先级）：

1. 会不会改变 retriever 的检索结果？会 → 不要做。
2. 会不会让竞品绕过 baseline retriever 联网？会 → 禁止。
3. 会不会让不同系统用不同检索配置？会 → 禁止。
4. 会不会影响 evaluator 需要的 `retrieved_docids`？会 → 必须显式修正。
5. 只是工程优化、不改变检索结果和输出格式 → 可以做。

## 0.1 ParallelMuse 是什么、和什么配合使用

- 仓库 `Alibaba-NLP/DeepResearch` 是 Tongyi Lab 的 monorepo；ParallelMuse 原始代码在 `WebAgent/ParallelMuse/` 下（`functionality_specified_partial_rollout.py` + `compressed_reasoning_aggregation.py`，**这两个原始文件禁止修改**）。
- ParallelMuse 是**纯推理时（test-time scaling）方法**，不训练模型。两阶段：
  - Stage 1（partial rollout）：每题采样预算 N 条轨迹 = 少量完整轨迹 + 在高不确定性步骤（按 token 级 PPL）分支复用前缀的部分轨迹 + 补充完整轨迹。
  - Stage 2（aggregation）：每条轨迹压缩成报告，再一次性聚合出每题 1 个最终答案。
- **它自带完整 ReAct 循环**（`rollout_single_traj` 内嵌，与 Tongyi 官方 ReAct 同一套 `<tool_call>` 协议），不需要外接任何 agent 框架。"配合使用"只发生在模型层：本实验用 OpenRouter 的 `openai/gpt-oss-20b`。
- 预算 8 的构成（发布代码默认参数）：`1 条初始完整轨迹 + 2 个分支点 × 3 次续写 + 1 条补充完整轨迹 = 8`。

## 0.2 已经完成的工作——执行 agent 的任务是"跑实验"，不是"写代码"

所有适配代码已写好、离线验证通过、并已提交到 fork 仓库 `git@github.com:Mmmmroy0806/DeepResearch.git` 的 **`deepsearch` 分支**（关键 commit：`acc046e` 适配主体、`a081a8f` 编排器、`e11ee7e` 健壮性修复）。开始前先确认在该分支上：

```bash
cd /Users/mmmroym/Downloads/huawei/DeepResearch && git checkout deepsearch
```

文件清单（均已存在，不要重写）：

| 文件 | 作用 |
| --- | --- |
| `inference/baseline_retriever.py` | deepsearch_edi baseline Milvus retriever 的最小移植（仅依赖 pymilvus+requests），检索参数锁死 |
| `inference/configs/bcp_baseline_retriever.json` | retriever 配置（唯一需要按环境修改的文件，见 §1） |
| `inference/test_bcp_alignment.py` | 与 deepsearch_edi 原 retriever 的 top-k chunk_id 对齐测试 |
| `WebAgent/ParallelMuse/bcp_partial_rollout.py` | Stage 1 适配（内嵌 ReAct + baseline retriever search-only 工具） |
| `WebAgent/ParallelMuse/bcp_aggregate.py` | Stage 2 适配（修复了原脚本两个 TypeError），输出 evaluator 格式 |
| `WebAgent/ParallelMuse/bcp_traj_to_eval.py` | 单轨迹 ReAct 对照组的格式转换器（可选条目用） |
| `WebAgent/ParallelMuse/run_react_rollouts.py` + `configs/react8_gpt_oss_20b.yaml` | "ReAct 独立跑 8 次"基线协议的 YAML 编排器（**不用于 ParallelMuse 条目**，见 §5.7） |
| `WebAgent/ParallelMuse/README_BCP.md` | 运行文档（与本方案一致，命令可直接复制） |

已离线验证过的内容（不需要重复验证）：三步流程的 `main()` 端到端 mock 仿真、每步 resume 幂等、错误轨迹恢复、无 logprobs 降级路径、PPL span 字符偏移映射、docid 并集提取、evaluator 输出格式。**唯一没验证的是需要真实外部服务的部分**（OpenRouter key、Milvus 可达性、provider 实际 logprobs 质量），这正是 §2 第 0 步冒烟的目的——它们都会在第一题就暴露，不会跑到一半才发现。

## 1. 环境前提（跑之前逐项确认）

1. **分支**：`deepsearch` 分支（见上）。
2. **Python 依赖**：`openai numpy json5 tqdm pymilvus requests pyyaml`（`pip install` 补齐即可；不需要 GPU，全部计算在 API 侧）。
3. **Milvus**：编辑 `inference/configs/bcp_baseline_retriever.json` 的 `milvus.uri` 指向实际 Milvus 服务（如有认证填 `token`）。**只允许改 `milvus.uri`/`token`，`retrieval` 段和 `collection_name` 一律不许动**：collection 必须是论文 baseline collection `browsecompplus_v2_baseline_512tok`（database `deepsearch_benchmarks`），top_k=3 / hybrid / baseline=true / add_instruction=true 是论文口径，hybrid 权重 (0.6, 0.4) 在代码里锁死（配置传其他值会直接报错，这是有意设计）。
4. **API key**：`export OPENROUTER_API_KEY=...`（embedding 和 LLM 共用这一个）。
5. **数据集**：BrowseComp-Plus 的 `topics-qrels/queries.tsv`（格式 `query_id<TAB>query`）。本机参考路径 `/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/topics-qrels/queries.tsv`，服务器上按实际路径替换。前 100 条通过 `--limit 100` 实现，**不要手工切分数据文件**。
6. **模型**：`openai/gpt-oss-20b`（已确认存在于 OpenRouter，131k context）。

## 2. 执行协议

统一变量（后续命令直接引用）：

```bash
cd /Users/mmmroym/Downloads/huawei/DeepResearch/WebAgent/ParallelMuse
export OPENROUTER_API_KEY=...
OUT=./bcp_results/pm_gpt_oss_20b
QA=/path/to/BrowseComp-Plus/topics-qrels/queries.tsv
M="openai/gpt-oss-20b"
EB='{"reasoning": {"enabled": true}, "provider": {"require_parameters": true}}'
```

**EB 里的 `provider.require_parameters` 不可省略**：OpenRouter 上 gpt-oss-20b 有 12 家 provider，只有 WandB/Novita/Parasail 支持全部所需参数（logprobs/top_logprobs/stop/presence_penalty）。不锁定会导致 logprobs 时有时无（破坏分支点检测的一致性），甚至路由到不支持 `stop` 的 provider（模型会越过 `<tool_response>` 幻觉工具结果，污染整条轨迹）。

### 第 0 步：对齐硬验收（一次性，若此前已做过可跳过）

在 deepsearch_edi 的运行环境里执行：

```bash
cd /Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi
.venv/bin/python /Users/mmmroym/Downloads/huawei/DeepResearch/inference/test_bcp_alignment.py
```

验收：输出 `ALIGNMENT OK: all top-k chunk_ids identical.`——证明适配 retriever 与论文 baseline 逐位一致。若 MISMATCH，按脚本提示排查 collection/embedding/mode/top_k，**不要继续往下跑**。（注：该脚本需要 deepsearch_edi 环境能 import `openjiuwen`；本机的 .venv 缺 `openjiuwen.core.foundation`，需在实际跑论文实验的环境执行。）

### 第 1 步：冒烟 + logprobs 探测（1 题）

```bash
python bcp_partial_rollout.py --qa_file_path $QA --limit 1 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 1

python -c "
import json
d=[json.loads(l) for l in open('$OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl')]
t=d[0]
print('termination:', t['termination'])
print('prediction:', t['prediction'][:100])
print('retrieved_docids:', t['retrieved_docids'][:5])
print('step_ppl samples:', [m.get('step_ppl') for m in t['rollout'] if m['role']=='assistant'][:3])
"
```

验收（四项全过才继续）：

- `termination` 为 `answer`（或至少不是 `llm_error_occurred`）；
- `retrieved_docids` 非空，且**不是** `xxx__数字` 的 chunk_id 格式（docid 与 chunk_id 是分开记录的，evaluator 要 docid）；
- 轨迹里的 tool_response 只包含 baseline retriever 的结果（有 `docid:`/`chunk_id:`/`score:` 行），**绝不能出现** Serper/Jina/Google/visit/URL 访问痕迹——出现即实现被用错，停止并检查是否误用了原始脚本；
- 记录这一题的 token 消耗（OpenRouter 后台可见），乘以 800（100 题 × 8 轨迹）估算总预算，确认可接受后再放开。

**分岔判定**：看 `step_ppl`——有数值 → 走第 2 步路线一（论文完整方法）；全是 `null` → 走路线二（trajectory-level 并行，也是论文报告过的合法设置）。已加 provider 锁定的情况下预期是有数值的。

### 第 2 步（路线一，step_ppl 有值）：完整方法三步

```bash
# Step A：每题 1 条初始完整轨迹
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 1

# Step B：不确定性引导部分 rollout，补到每题 8 条（发布代码默认参数）
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode tool_call_ppl \
  --initial_rollout_num 1 --partial_sampling_topk 2 \
  --partial_sampling_times_per_pos 3 --sampling_budget 8

# Step C：压缩报告 + 聚合（聚合模型 = 同一个 gpt-oss-20b，这是论文口径，不要换模型）
python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_tool_call_ppl_2_1_3.jsonl \
  --output-dir $OUT/aggregated \
  --llm-model $M --extra-body "$EB" --store-reports
```

### 第 2 步（路线二，step_ppl 全 null）：trajectory-level 并行

```bash
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 8

python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl \
  --output-dir $OUT/aggregated \
  --llm-model $M --extra-body "$EB" --store-reports
```

两条路线只能选一条，选了就全程一致；论文里必须注明用的是哪个变体（uncertainty-guided 还是 trajectory-level）。

### 第 3 步：中途检查点（建议在 Step A 跑完后做一次）

```bash
python -c "
import json, collections
f='$OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl'
c=collections.Counter(); term=collections.Counter()
for l in open(f):
    r=json.loads(l); term[r['termination']]+=1
    if r['termination']!='llm_error_occurred': c[r['question']]+=1
print('terminations:', dict(term))
print('questions with valid rollout:', len(c))
"
```

- `llm_error_occurred` 占比高 → 检查 key 余额/限流，然后**直接重跑同一条命令**（错误轨迹不计入预算，会自动补）；
- 任何步骤中断（断网/Ctrl-C/进程被杀）的恢复方式都是：**原命令重跑**，已完成部分自动跳过。

### 第 4 步：评测

`$OUT/aggregated/` 下每题一个 `run_*.json`，就是 BrowseComp-Plus evaluator 的输入格式（含 `query_id`/`status`/`retrieved_docids`/`result[0].output`）。用 BrowseComp-Plus 仓库（本机参考路径 `/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus`）的 evaluator 对该目录评测，qrels 在其 `topics-qrels/` 下。

最终验收：

- `aggregated/` 恰好 100 个文件（每 query_id 一个，无重复）；
- 每个文件 `retrieved_docids` 非空、非 chunk_id 格式、语义是**该题全部 8 条并行轨迹的并集**（这是诚实口径：聚合答案可能引用任何一条轨迹的证据）;
- `metadata` 含 `method`/`model`/`n_trajectories: 8`/`merge_num`/retriever 溯源（collection、mode、top_k、baseline、add_instruction）;
- `status: completed` 的比例正常（个别 `no_valid_answer` 可接受，指该题 8 条轨迹全都没给出答案）。

## 3. 已经确认的设计决策（不要重新讨论，也不要"优化"）

1. **采样参数用 ParallelMuse 自己的**：temperature 0.6 / top_p 0.95 / presence_penalty 1.1（方法自带 ReAct 循环的原始参数），不是 deepsearch_edi 配置里的值。
2. **聚合模型 = agent 模型**（同一个 gpt-oss-20b），Stage 2 不换更强的模型。
3. **retriever 参数锁死**：模型只能通过 tool arguments 传 query 字符串；top_k/mode/collection/instruction 全部来自配置文件，hybrid 权重 (0.6,0.4) 硬编码。这是实验控制，不是可调项。
4. **search-only**：唯一工具是 `search`。ParallelMuse 原版的 `visit` 已被移除，prompt 中也不声明。模型请求其他工具会收到 "does not exist" 回复，这是预期行为。
5. **每条轨迹的 tool 返回格式**：编号列表 + `docid:`/`chunk_id:`/`score:` 行。与 deepsearch_edi 自家 agent 看到的格式不同，但检索结果本身完全一致（方案层面已确认接受）。
6. **前 100 条用 `--limit 100`**：取 queries.tsv 的前 100 行，顺序即文件顺序。

## 4. 已经排除的方案（不要走回头路）

- ❌ 使用 ParallelMuse 原始脚本 + 自补 tools/（它们依赖联网 Serper/Jina，且聚合脚本有两个 TypeError 根本跑不通）；
- ❌ 使用仓库根目录 `inference/tool_search.py`（Serper 联网版）或 `tool_visit.py`；
- ❌ 使用阿里云百炼的 deepsearch 应用（黑盒 agent，检索不可替换，违反实验控制）；
- ❌ import 整个 `openjiuwen_deepsearch` 包到竞品代码（依赖冲突，且已用最小移植替代）；
- ❌ 单独部署 retriever HTTP service（retriever 逻辑作为进程内 Python adapter 存在）；
- ❌ 本地重建 Milvus 索引（用服务器上现成的 collection）；
- ❌ LEGO retriever（本任务只用 baseline，LEGO 是另一个任务）；
- ❌ AgentFold（另一个竞品，权重在 ModelScope `iic/AgentFold-30B-A3B-Preview`，不在本任务范围内）。

## 5. 常见坑（全部来自前期实际排查，遇到时按此处理）

1. **`SERPER_KEY_ID`/`JINA_API_KEYS` 相关报错** → 说明误用了原始联网脚本。BCP 模式只能跑 `bcp_partial_rollout.py`/`bcp_aggregate.py`。
2. **retriever 构造时报 collection 不存在** → `milvus.uri` 指错了或没连上服务器。这是启动即报的错，不是代码 bug。
3. **把 chunk_id 当 docid** → Milvus 主键是 `{docid}__{chunk_idx}`；evaluator 要 docid。适配代码已分开记录，如自查输出发现 `retrieved_docids` 里有 `__数字` 结尾的条目，立即停止并上报，不要自行改评测数据。
4. **logprobs 时有时无** → 忘了 EB 里的 `provider.require_parameters`。加上重跑；已产生的混合轨迹建议废弃该输出目录重来（分支点一致性被破坏）。
5. **max_tokens 相关 4xx** → 代码会自动减半重试（最低 1024），无需干预。
6. **`llm_error_occurred` 轨迹** → 不计入预算，重跑同一命令自动补齐。不要手工编辑 jsonl 文件。
7. **不要用 `run_react_rollouts.py`（8 环境编排器）跑 ParallelMuse 条目**——那是"ReAct 独立跑 8 次求均值"的基线协议，与 ParallelMuse 方法（8 条轨迹聚合成 1 个答案）是不同的实验条目。除非用户明确要求补 "ReAct (avg of 8)" 基线行，否则本任务不用它。
8. **成本**：每题最多 8 条轨迹 × 100 轮 LLM 调用 + 8 次报告 + 1 次聚合。务必先做第 1 步的单题成本估算，异常昂贵（单题 > 预估一个数量级）时停止并上报，不要硬跑。
9. **Step B 的断言 `Initial rollouts are not sufficient`** → Step A 没跑完或错误轨迹太多，先重跑 Step A（见第 3 步检查点）。
10. **输出文件名是自动生成的**（含数据集名、模型名、采样参数），Step C 的 `--rollout-file` 必须指向 Step B 实际产出的文件名，路线一是 `..._1_tool_call_ppl_2_1_3.jsonl`，路线二是 `..._1_none_initial_rollout.jsonl`。

## 6. 需要留给论文的证据（跑完检查一遍）

- `aggregated/run_*.json` 的 metadata 自动包含：method、model、n_trajectories、merge_num、retriever 名称/collection/mode/top_k/baseline/add_instruction；
- 每题的 `tool_call_counts.search`（总检索次数）；
- 手工补记：选择的路线（uncertainty-guided / trajectory-level）及原因（step_ppl 探测结果）、总 token 消耗、provider 锁定配置；
- 论文表述可参考：*"ParallelMuse is evaluated with the same fixed BrowseComp-Plus baseline retriever as our system (same Milvus collection, embedding, query instruction, hybrid mode with WeightedRanker(0.6,0.4), top-k, and baseline chunk-return policy). Its original web search and browsing tools are replaced by a single search tool backed by this retriever; no web access is enabled. We use the released sampling configuration (budget 8: 1 initial rollout, 2 branch points × 3 continuations, 1 supplementary rollout) with gpt-oss-20b as the backbone, and the same model for report compression and aggregation."*

## 7. 禁止事项（红线）

1. 禁止修改 `/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi` 下任何文件（论文投稿代码）。
2. 禁止修改 ParallelMuse 原始脚本（`functionality_specified_partial_rollout.py`、`compressed_reasoning_aggregation.py`）。
3. 禁止修改 `inference/configs/bcp_baseline_retriever.json` 的 `retrieval` 段和 `collection_name`（只许改 `milvus.uri`/`token`）。
4. 禁止启用任何联网工具（visit/Serper/Jina/Google/Scholar）。
5. 禁止把明文 API key 写进任何文件或提交到 git（配置里用 `${OPENROUTER_API_KEY}` 占位符，运行时从环境变量展开）。
6. 如需修 bug，只允许改 `bcp_*` 前缀的适配文件，并保持"不改变检索结果、不改变输出格式、不改变方法逻辑"三不变。
