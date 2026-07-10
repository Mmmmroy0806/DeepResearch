# ParallelMuse 竞品实验执行方案（BrowseComp-Plus + baseline retriever + gpt-oss-20b）

本文档是给执行 agent 的完整任务说明。执行 agent 不具备前序讨论上下文，因此本文档不只告诉你"做什么"，更重要的是告诉你**每个决策背后的原因**。请先把 §0 读完再动手：这个实验的所有约束都服务于一个科学目的，理解了目的，你才能在遇到文档没覆盖的情况时做出正确判断，而不是瞎猜。

**一句话目标：在固定 baseline retriever 的前提下，用 ParallelMuse 方法（预算 8）+ gpt-oss-20b 跑 BrowseComp-Plus 前 100 条 query，产出可供 BrowseComp-Plus evaluator 评测的结果，作为 deepsearch_edi 论文的竞品条目。**

---

## 0. 背景：这个实验为什么存在、为什么长这样

### 0.1 论文的处境和竞品实验的目的

用户正在投稿 `deepsearch_edi` 论文（代码在 `/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi`）。论文提出了一套 deepsearch 方法，在 BrowseComp-Plus（下称 BCP）等 benchmark 上评测。审稿人必然会问："你们的方法比现有的 deep research 系统好在哪？"——所以需要竞品对比实验。

竞品之一是 **ParallelMuse**（Tongyi Lab，arxiv 2510.24698）：一个推理时并行思考方法。选它的原因：它是 2025 年末最新的 test-time scaling 代表工作，且开源了代码（在 `Alibaba-NLP/DeepResearch` monorepo 的 `WebAgent/ParallelMuse/` 下）。

搭载模型选 `openai/gpt-oss-20b` 的原因：deepsearch_edi 论文自己的实验矩阵里就有 gpt-oss-20b 这一档（见参考配置 `deepsearch_edi/benchmarking/openai_config_bcp/openai_gpt_oss_20b_react_baseline_512tok_verbatim_0521.json`），竞品用同一个模型，"方法之间的对比"才不会被"模型之间的差异"污染。ParallelMuse 本身是模型无关的推理时方法（论文在多个开源 agent 上验证过），换模型不破坏方法本体。

### 0.2 为什么必须"固定 retriever"——这是全文档一切约束的根源

BCP 和联网版 BrowseComp 的本质区别：BCP 是**固定语料**检索 benchmark，它的设计目的就是把"检索器的贡献"和"agent/LLM 的贡献"隔离开。deepsearch_edi 论文的实验设计是：所有系统（自己的方法和全部竞品）都用**同一个 baseline retriever**（同一个 Milvus 索引、同一个 embedding 模型和 query instruction、同一种 hybrid 检索、同一个 top_k、同一种 chunk 返回策略），这样表格里任何两行的分数差异都只能归因于 agent/方法本身。

由此推出实验要回答的问题**不是**"ParallelMuse 自己联网能做到什么"，而是：

```text
在使用与 deepsearch_edi 完全相同的 BCP baseline retriever 的前提下，
ParallelMuse 这个方法（搭载 gpt-oss-20b）的表现如何？
```

只要有任何一条证据来自 baseline retriever 之外（联网搜索、网页访问、别的索引、别的 top_k），这一行数据就作废——不是"效果打折"，是**科学上不可用**，因为差异无法归因了。

遇到文档没覆盖的情况时，用这五条原则判断（按优先级）：

1. 会不会改变 retriever 的检索结果？会 → 不要做。
2. 会不会让竞品绕过 baseline retriever 拿到语料外的信息？会 → 禁止。
3. 会不会让不同系统用不同检索配置？会 → 禁止。
4. 会不会影响 evaluator 需要的 `retrieved_docids`？会 → 必须显式修正。
5. 只是工程优化、不改变检索结果和输出格式 → 可以做。

### 0.3 ParallelMuse 是什么、为什么它不需要"配合别的框架"

前期讨论中曾有一个疑问：ParallelMuse 是不是要配合 ReAct 框架或 Tongyi DeepResearch 一起用？读源码后的结论是**不需要**，理由如下：

- ParallelMuse 是纯推理时方法，两阶段：**Stage 1**（partial rollout）按每题的采样预算 N 生成 N 条轨迹——少量完整轨迹 + 在"高不确定性步骤"（用 token 级 PPL 度量）分支复用前缀的部分轨迹 + 补充完整轨迹；**Stage 2**（aggregation）把每条轨迹压缩成报告，再一次聚合出每题 1 个最终答案。
- Stage 1 的 `rollout_single_traj` 函数**本身就是一个完整的 ReAct 循环**（system prompt 声明 `<tools>`，模型输出 `<think>`+`<tool_call>`，循环执行工具拼 `<tool_response>`，直到 `<answer>`），且与 Tongyi 官方 `inference/react_agent.py` 的协议逐字相同。所以"ReAct"已经内嵌在方法里，唯一的外部依赖是一个 base model。
- 预算 8 的构成（发布代码的默认参数，本实验沿用以保持"复现论文方法"的口径）：`1 条初始完整轨迹 + 2 个最高不确定性分支点 × 3 次续写 + 1 条补充完整轨迹 = 8`。分支续写复用前缀是这个方法的效率贡献（省 token），聚合是质量贡献。

另外注意：Tongyi 仓库 README 提到的 IterResearch "Heavy mode" 是**另一种** test-time scaling 范式、另一篇论文，代码未开源，与 ParallelMuse 无关，不要混淆。

### 0.4 代码已经全部写好——你的任务是"跑实验"，不是"写代码"

为什么强调这一点：适配工作经历了多轮实现、review、修复（包括两轮"修复本身引入新问题再修复"的迭代），现在的代码状态是经过对抗性审查的。执行 agent 如果绕开现成代码自己写，等于把所有已经踩过的坑重新踩一遍。

所有代码在 fork 仓库 `git@github.com:Mmmmroy0806/DeepResearch.git` 的 **`deepsearch` 分支**（关键 commit：`acc046e` 适配主体、`e11ee7e` 健壮性修复与 provider 锁定）。开始前确认分支：

```bash
cd /Users/mmmroym/Downloads/huawei/DeepResearch && git checkout deepsearch
```

文件清单及各自存在的原因：

| 文件 | 作用 | 为什么需要它 |
| --- | --- | --- |
| `inference/baseline_retriever.py` | deepsearch_edi baseline retriever 的最小移植 | 不能 import 整个 openjiuwen 包（依赖冲突，见 §4），所以做了逐参数对齐的最小复制；已通过 8/8 参数机械对照 |
| `inference/configs/bcp_baseline_retriever.json` | retriever 配置 | 所有竞品条目共用这一份配置 → 检索面天然一致 |
| `inference/test_bcp_alignment.py` | 与原 retriever 的 chunk_id 对齐测试 | "用了同一套 retriever"的唯一硬证明（§2 第 0 步） |
| `WebAgent/ParallelMuse/bcp_partial_rollout.py` | Stage 1 适配 | 原始脚本依赖联网工具且假设 Qwen tokenizer/vLLM logprobs，直接跑不通（详见 §4、§5） |
| `WebAgent/ParallelMuse/bcp_aggregate.py` | Stage 2 适配 | 原始聚合脚本有两个多传参数的 TypeError，发布状态跑不通；此外要把输出转成 evaluator 格式 |
| `WebAgent/ParallelMuse/bcp_traj_to_eval.py` | 单轨迹→evaluator 格式转换器 | 可选的 "ReAct 对照组" 用（§3.7），本任务主线不用 |
| `WebAgent/ParallelMuse/run_react_rollouts.py` + `configs/react8_gpt_oss_20b.yaml` | 8 环境 YAML 编排器 | **不用于本任务**（§5.7 解释为什么容易混、区别是什么） |
| `WebAgent/ParallelMuse/README_BCP.md` | 运行文档 | 与本方案一致，命令可直接复制 |

已经离线验证过、**不需要重复验证**的内容：三步流程的 `main()` 用 mock LLM/retriever 端到端仿真通过（含每题精确 8 条的预算核算）、每步 resume 幂等、错误轨迹恢复、无 logprobs 降级路径、PPL span 定位、docid 并集提取、evaluator 输出格式。唯一没验证的是需要真实外部服务的部分（OpenRouter key、Milvus 可达性、provider 实际 logprobs 质量）——这正是 §2 第 1 步冒烟的目的，它们的共同特点是**第一题就会暴露，不会跑到一半才发现**。

---

## 1. 环境前提（每项都说明为什么）

1. **分支 `deepsearch`**：适配代码只在这个分支上，main 分支是 Tongyi 官方原始代码。
2. **Python 依赖**：`pip install openai numpy json5 tqdm pymilvus requests pyyaml`。不需要 GPU——LLM 和 embedding 全在 OpenRouter API 侧，本地只做异步调度和 Milvus 查询，所以这个实验可以在任何一台能联网的机器上发起。
3. **Milvus**：编辑 `inference/configs/bcp_baseline_retriever.json` 的 `milvus.uri` 指向实际服务（有认证则填 `token`）。**只许改 `uri`/`token` 两项**。为什么其他项不许动：`collection_name: browsecompplus_v2_baseline_512tok`（database `deepsearch_benchmarks`）是论文 baseline 用的那个索引，`retrieval` 段（top_k=3 / hybrid / baseline=true / add_instruction=true）是论文 baseline 的检索口径——改任何一项都违反 §0.2 的原则 3。hybrid 权重 (0.6, 0.4) 在代码里硬编码且配置传其他值会直接报错，这是有意设计：deepsearch_edi 原代码里这个权重就是硬编码的，不存在合法变体，开放成可调项只会制造悄悄偏离 baseline 的入口。
4. **API key**：`export OPENROUTER_API_KEY=...`。embedding（qwen3-embedding-8b）和 LLM（gpt-oss-20b）共用这一个 key。配置文件里写的是 `${OPENROUTER_API_KEY}` 占位符、运行时从环境变量展开——所以 key 永远不落盘、不进 git。
5. **数据集**：BCP 的 `topics-qrels/queries.tsv`（`query_id<TAB>query`）。本机参考路径 `/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/topics-qrels/queries.tsv`，服务器上按实际路径替换。**前 100 条用 `--limit 100` 实现，不要手工切分文件**——因为断点续跑和进度统计都以完整文件 + limit 为基准，手工切出来的文件会让"哪 100 条"变得不可追溯。
6. **模型 id**：`openai/gpt-oss-20b` 已确认存在于 OpenRouter（131k context），不需要再验证存在性；但它有 12 家 provider，参数支持参差不齐，这是 §2 的 EB 配置存在的原因。

## 2. 执行协议

统一变量：

```bash
cd /Users/mmmroym/Downloads/huawei/DeepResearch/WebAgent/ParallelMuse
export OPENROUTER_API_KEY=...
OUT=./bcp_results/pm_gpt_oss_20b
QA=/path/to/BrowseComp-Plus/topics-qrels/queries.tsv
M="openai/gpt-oss-20b"
EB='{"reasoning": {"enabled": true}, "provider": {"require_parameters": true}}'
```

**EB 的两个字段各自的原因**：

- `reasoning.enabled`：gpt-oss 是推理模型，deepsearch_edi 自己跑 gpt-oss-20b 的配置里就开了这个开关（见 §0.1 的参考配置）。竞品不开的话，模型行为与 deepsearch_edi 实验里的同名模型不对等，违反变量控制。
- `provider.require_parameters`：**不可省略**。查证过 OpenRouter 上 gpt-oss-20b 的 12 家 provider 中只有 WandB/Novita/Parasail 支持全部所需参数（logprobs/top_logprobs/stop/presence_penalty）。不锁定的后果有两级：轻则 logprobs 时有时无——Stage 1 的分支点检测就会对一部分轨迹生效、另一部分失效，方法既不是论文的 uncertainty-guided 也不是干净的纯并行，这行数据没法写进论文；重则路由到不支持 `stop` 的 provider（如 SiliconFlow/Google），模型生成时不会在 `<tool_response>` 处停下，而是自己编造工具返回结果继续写——整条轨迹被幻觉污染且**表面上看不出来**。

### 第 0 步：对齐硬验收（一次性，若已做过可跳过）

**为什么要做**：整个实验的科学前提是"竞品用了和论文完全相同的 retriever"。适配 retriever 是从 deepsearch_edi 逐参数移植的复制品，逐字符对照过 8 个关键参数（instruction/embedding payload/dense ef/sparse BM25/WeightedRanker/output_fields/baseline limit/embed dims），但"代码看起来一样"和"检索结果逐位一致"之间仍需一个运行时证明——这就是对齐测试：同 query、同 config 下，两边返回的 top-k chunk_id 必须完全一致（含顺序）。

```bash
cd /Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi
.venv/bin/python /Users/mmmroym/Downloads/huawei/DeepResearch/inference/test_bcp_alignment.py
```

验收：输出 `ALIGNMENT OK: all top-k chunk_ids identical.`。若 MISMATCH，按脚本提示排查（通常是 collection 指错、embedding base_url 不一致），**修好之前禁止往下跑**——后面跑得再多，retriever 不对齐就全部作废。

注意：此脚本需在能 import `openjiuwen` 的 deepsearch_edi 运行环境执行（开发机的 .venv 缺 `openjiuwen.core.foundation`，要在实际跑论文实验的环境做）。

### 第 1 步：冒烟 + logprobs 探测（1 题）

**为什么要做**：三个目的。① 所有外部依赖（key、Milvus、provider）在第一题全部过一遍，避免配置问题在全量运行中段爆发；② 探测锁定 provider 后 logprobs 的实际质量，决定 §2 走哪条路线；③ 拿到单题真实 token 成本，估算总预算——这个实验每题最多 8 条轨迹 × 100 轮 LLM 调用 + 8 次报告 + 1 次聚合，成本约是单轨迹 ReAct 的 10 倍，先算账再放开是负责任的做法。

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

验收四项（全过才继续）及各自在防什么：

- `termination` 为 `answer`（不是 `llm_error_occurred`）→ 证明 key/模型/路由通了；
- `retrieved_docids` 非空且**没有** `xxx__数字` 形态的条目 → evaluator 用 docid 算 recall，chunk_id 混进去会让 recall 虚假偏低（Milvus 主键是 `{docid}__{chunk_idx}`，两者容易混）；
- 轨迹里的 tool_response 只含 baseline retriever 的结果（有 `docid:`/`chunk_id:`/`score:` 行），**绝无** Serper/Jina/Google/visit/URL 痕迹 → 出现即说明跑的不是适配脚本而是原始联网脚本，立即停止（§0.2：语料外证据 = 数据作废）;
- 记录该题 token 消耗 ×800 估算总成本，若单题成本比预期高一个数量级，停止并上报，不要硬跑。

**路线分岔判定**：看 `step_ppl`。有数值 → 路线一（论文完整方法）；全 `null` → 路线二。已加 provider 锁定时预期有数值；若锁定后仍为 null，说明这三家 provider 的 logprobs 实现有变，此时路线二不是妥协——它对应论文自己报告过的 trajectory-level 并行消融设置，方法学上完全合法，只是论文里要如实注明变体。

### 第 2 步（路线一）：完整方法三步

**为什么是三步、为什么这个顺序**：Step B 要从 Step A 的初始轨迹里找高不确定性分支点（没有初始轨迹就没有分支的对象）；Step C 要等每题 8 条轨迹齐了才能聚合。步与步之间串行，每步内部由 asyncio 并发（`--max_llm_workers` 控制），所以**不需要**任何外部并行编排。

```bash
# Step A：每题 1 条初始完整轨迹
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 1

# Step B：在初始轨迹的 2 个最高 tool_call-PPL 步骤上各分支续写 3 次（复用前缀），
#         再补 1 条完整轨迹，凑满每题 8 条。参数是发布代码的默认值，沿用以保持
#         "复现论文方法"的口径，不要调。
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode tool_call_ppl \
  --initial_rollout_num 1 --partial_sampling_topk 2 \
  --partial_sampling_times_per_pos 3 --sampling_budget 8

# Step C：每条轨迹压缩成报告，再聚合出每题 1 个最终答案
python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_tool_call_ppl_2_1_3.jsonl \
  --output-dir $OUT/aggregated \
  --llm-model $M --extra-body "$EB" --store-reports
```

### 第 2 步（路线二）：trajectory-level 并行

```bash
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 8

python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl \
  --output-dir $OUT/aggregated \
  --llm-model $M --extra-body "$EB" --store-reports
```

两条路线只能选一条且全程一致——中途换路线会让不同题目的轨迹生成方式不同，这行数据就没有统一的方法学描述了。

### 第 3 步：中途检查点（Step A 跑完后做一次）

**为什么**：Step B 开头会断言每题都有足够的可用初始轨迹（错误轨迹不算数）。与其让断言在 Step B 才报错，不如 Step A 完成后主动检查：

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

- `llm_error_occurred` 多 → 通常是限流或余额，处理后**原命令重跑**即可：错误轨迹被设计成不计入预算（这是修过的一个隐患：原逻辑下错误轨迹占坑，一次瞬时 API 故障就会让 Step B 永久断言失败），重跑会自动补齐。
- 任何步骤被中断（断网/Ctrl-C/进程被杀）的恢复方式统一是**原命令重跑**：Step A/B 有按题 resume，Step C 按 query_id 跳过已完成的。**不要手工编辑 jsonl 文件**——resume 逻辑靠逐行计数，手工增删会破坏预算核算。

### 第 4 步：评测与最终验收

`$OUT/aggregated/` 下每题一个 `run_*.json`，即 BCP evaluator 的输入。用 BCP 仓库（本机参考 `/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus`）的 evaluator 评测，qrels 在其 `topics-qrels/` 下。

最终验收清单：

- `aggregated/` 恰好 100 个文件，query_id 无重复；
- 每个文件 `retrieved_docids` 非空、非 chunk 格式，语义是**该题 8 条并行轨迹检索结果的并集**。为什么取并集：聚合答案可能引用任何一条轨迹的证据，只报某一条轨迹的 docids 会低估证据面，是不诚实的口径；
- `metadata` 含 method / model / `n_trajectories: 8` / merge_num / retriever 溯源（collection、mode、top_k、baseline、add_instruction）——这些是论文可复现性证据，自动生成，检查存在即可；
- `status: completed` 占绝大多数（个别 `no_valid_answer` 可接受，含义是该题 8 条轨迹全都没产出答案）。

## 3. 已确认的设计决策及其理由（不要重新讨论，理由已经过多轮审查）

1. **采样参数用 ParallelMuse 自己的（0.6/0.95/1.1），不用 deepsearch_edi 配置里的**。原因：这些参数属于"被评测的方法"的一部分——ParallelMuse 的 ReAct 循环连同它的采样参数一起构成论文发布的方法，换参数后跑出来的就不是"ParallelMuse"了，与其论文数字的可比性也没了。
2. **聚合模型 = agent 模型（同一个 gpt-oss-20b）**。原因：论文口径如此；换更强的聚合模型会把"方法的贡献"和"更强裁判的贡献"混在一起，竞品分数虚高或虚低都会误导对比。
3. **retriever 参数在构造时锁死，模型只能传 query 字符串**。原因：§0.2 原则 3。如果模型能通过 tool arguments 控制 top_k/mode，不同轨迹的检索配置就不一致了。
4. **search-only，原版的 visit 工具已移除且 prompt 不声明**。原因：BCP 是固定语料 benchmark，visit 访问 live web 拿到的是语料外证据（§0.2）。模型若请求不存在的工具会收到 "does not exist" 回复——这是预期行为，不是 bug，不要"好心"把 visit 加回来。
5. **工具返回格式是编号列表 + `docid:`/`chunk_id:`/`score:` 行**。与 deepsearch_edi 自家 agent 看到的 JSON snippet 格式不同——这是方案层面确认过接受的差异（检索结果本身逐位一致，只是呈现格式遵循 Tongyi 系工具的习惯）。已知的开放小问题：这个格式让竞品比 deepsearch_edi 的 agent 多看到 score 数值，若用户后续要求对齐信息暴露量，删 `score:` 行即可（在 `bcp_partial_rollout.py` 的 `_format_query_result`），除此之外不要改格式。
6. **前 100 条、预算 8**。前 100 条是用户的成本控制决策（先跑一批看结果再决定是否全量）；预算 8 是 ParallelMuse 发布代码的默认预算，也与 deepsearch_edi 其他实验的 rollout 数量级可比。
7. **可选条目 "ReAct 对照组"**：用 `--sampling_budget 1` 的 Step A 输出 + `bcp_traj_to_eval.py` 转格式，可以得到"同模型 + 单轨迹 ReAct + 同 retriever"的基线行。它存在的原因：ParallelMuse 是预算依赖的方法，审稿人可能要求预算对齐的锚点。**但本任务的主线交付是 ParallelMuse 条目**，对照组是否跑、跑不跑 8 次求均值，是用户画表时的决策，没有用户明确指示不要自行扩大范围。

## 4. 已排除的方案及排除原因（不要走回头路）

- ❌ **直接跑 ParallelMuse 原始脚本 + 自补 tools/**：原始 `tools/` 目录是空的（只有 .gitkeep），其期望的 Search/Visit 是 async 接口且依赖联网 Serper/Jina（违反 §0.2）；且原始聚合脚本有两个多传参数的 TypeError，发布状态本身跑不通。
- ❌ **使用仓库根目录 `inference/tool_search.py`（Serper）或 `tool_visit.py`（Jina）**：联网工具，违反固定语料前提。如果运行日志里出现 `SERPER_KEY_ID` 或 `JINA_API_KEYS` 相关报错，说明误用了这些文件。
- ❌ **阿里云百炼的 deepsearch 应用/API**：查证过它是按 agent_id 调用的黑盒应用，内部自己联网检索，无法替换检索源——违反原则 2 和 3。
- ❌ **import 整个 `openjiuwen_deepsearch` 包**：依赖冲突（openjiuwen/pymilvus 版本、Python path），且会让竞品代码反向依赖论文代码。已用逐参数对齐的最小移植替代，对齐性由第 0 步测试保证。
- ❌ **单独部署 retriever HTTP service**：用户明确过最终运行形态是"竞品代码 + 服务器上的 Milvus"，retriever 逻辑作为进程内 Python adapter 存在，不额外起服务。
- ❌ **本地重建 Milvus 索引**：索引构建是独立的大工程且有不确定性（embedding 批处理、chunk 边界），用服务器上现成的论文 collection 才能保证"同一个索引"。
- ❌ **LEGO retriever**：论文的另一个检索器，属于后续另一个任务，本任务只用 baseline，不要引入、预留或混合任何 LEGO 逻辑。
- ❌ **AgentFold**：另一个竞品（权重在 ModelScope `iic/AgentFold-30B-A3B-Preview`，需要 GPU 自部署），不在本任务范围内。

## 5. 常见坑：现象、成因、处理

1. **报错提到 SERPER/JINA** → 成因：误用了原始联网脚本。处理：确认跑的是 `bcp_partial_rollout.py`/`bcp_aggregate.py`，检查有没有人改过 import。
2. **retriever 构造时报 collection 不存在 / 连接失败** → 成因：`milvus.uri` 没指到服务器。这是启动即报的错（有意设计成 fail-fast），不是代码 bug。
3. **`retrieved_docids` 里出现 `__数字` 结尾的条目** → 成因：docid 与 chunk_id 混淆（主键格式 `{docid}__{chunk_idx}`）。适配代码已分开记录，正常不会发生；若发生，是 bug，停止并上报，**不要自行改评测数据**。
4. **step_ppl 时有时无（同一次运行里混合）** → 成因：EB 里漏了 `provider.require_parameters`，路由漂移。处理：补上后**废弃该输出目录重跑**——混合轨迹的分支点检测不一致，不能与干净数据拼在一起。
5. **max_tokens 相关 4xx** → 代码会自动减半重试（下限 1024），无需干预；这是 OpenRouter 部分 provider 对长输出的限制。
6. **`llm_error_occurred` 轨迹** → 成因：10 次指数退避重试后仍失败（限流/余额/网络）。处理：解决根因后原命令重跑，自动补齐。设计上这类轨迹不计入预算，就是为了让"重跑"永远是安全且充分的恢复手段。
7. **不要用 `run_react_rollouts.py` 跑本任务** → 那是"ReAct 独立跑 8 次"的基线协议：8 次互相独立、每次每题 1 个答案、出 8 套评测文件，用于算均值/方差。ParallelMuse 是"8 条相关轨迹（6 条复用前缀）聚合成每题 1 个答案、出 1 套评测文件"。两者轨迹间关系、输出语义、论文表格里的行都不同。看到"8"就用编排器是最容易犯的混淆。
8. **Step B 报 `Initial rollouts are not sufficient`** → 成因：Step A 没跑完或错误轨迹过多。处理：回到第 3 步检查点，重跑 Step A。
9. **Step C 的 `--rollout-file` 文件名** → 文件名由脚本按（数据集、模型、采样参数）自动生成：路线一是 `queries_openai_gpt-oss-20b_1_tool_call_ppl_2_1_3.jsonl`，路线二是 `queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl`。指错文件不会报错但聚合的是错的轨迹集，务必对准。
10. **成本失控** → 第 1 步的单题估算是唯一的事前防线；运行中若发现消耗远超估算（例如模型陷入超长循环），停下来看轨迹里是否在重复无效搜索，上报而不是调参数硬压。

## 6. 需要留给论文的证据

- `aggregated/run_*.json` 的 metadata 自动记录：method、model、n_trajectories、merge_num、retriever 溯源（不含任何 key）。为什么记这些：竞品实验最容易被质疑的就是"你们是不是没给竞品用同样的检索"，metadata 就是逐题的自证。
- 需手工补记的三项：最终选择的路线（uncertainty-guided / trajectory-level）及依据（第 1 步 step_ppl 探测结果）、总 token 消耗、EB 完整内容。
- 论文方法段可用的表述：*"ParallelMuse is evaluated with the same fixed BrowseComp-Plus baseline retriever as our system (same Milvus collection, embedding model and query instruction, hybrid retrieval with WeightedRanker(0.6, 0.4), top-k, and baseline chunk-return policy). Its original web search and browsing tools are replaced by a single search tool backed by this retriever; no web access is enabled. We use the released sampling configuration (budget 8: 1 initial rollout, 2 branch points × 3 continuations, 1 supplementary rollout) with gpt-oss-20b as the backbone, and the same model for report compression and aggregation."*

## 7. 红线（违反任何一条都会使实验数据或论文代码受损）

1. **禁止修改 `/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi` 下任何文件**——这是正在投稿的论文代码，竞品实验不允许反向污染它。
2. **禁止修改 ParallelMuse 原始脚本**（`functionality_specified_partial_rollout.py`、`compressed_reasoning_aggregation.py`）——保留原样是"我们复现的是官方发布的方法"这一声明的物证。
3. **禁止修改 retriever 配置的 `retrieval` 段和 `collection_name`**（只许改 `milvus.uri`/`token`）——§0.2 原则 3。
4. **禁止启用任何联网工具**——§0.2 原则 2；一条语料外证据就作废整行数据。
5. **禁止把明文 key 写进文件或提交 git**——占位符 + 环境变量机制已就位，没有理由绕过。
6. **如需修 bug，只允许改 `bcp_*` 前缀的适配文件**，且守住三不变：不改变检索结果、不改变输出格式、不改变方法逻辑。修完先跑第 1 步冒烟再继续。
