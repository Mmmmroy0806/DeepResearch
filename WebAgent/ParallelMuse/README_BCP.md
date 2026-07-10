# ParallelMuse × BrowseComp-Plus（deepsearch_edi baseline retriever）

在固定 baseline retriever 的前提下，用 ParallelMuse 两阶段方法（不确定性引导的
并行部分 rollout + 压缩报告聚合）评测任意 OpenAI 兼容模型（OpenRouter 的
deepseek-v4-flash、或本地 vLLM 服务等）。原始脚本
`functionality_specified_partial_rollout.py` / `compressed_reasoning_aggregation.py`
未改动；BCP 版为 `bcp_partial_rollout.py` / `bcp_aggregate.py`。

## 与原版的差异

| 方面 | 原版 | BCP 版 |
| --- | --- | --- |
| 工具 | 联网 search + visit（tools/ 目录为空，需自补） | 仅 search，走 `inference/baseline_retriever.py`（与 Tongyi ReAct 竞品实验同一 retriever/config） |
| 模型 | 本地 vLLM 起 Tongyi-DeepResearch | `--llm-model` + `--llm-base-url`，默认 OpenRouter |
| logprobs | 必需，拿不到直接崩 | 可选；provider 不返回时该步 `step_ppl=None`，分支点退化为均匀间隔的 tool-call 步（确定性），或建议直接用 `--partial_sampling_mode none` |
| PPL 区间定位 | 假设 `<tool_call>` 等是单 token（仅 Qwen 成立） | 字符偏移映射到 token 边界，任意 tokenizer 可用 |
| think 内容 | 模型在 content 里输出 `<think>` | 兼容 OpenRouter reasoning 字段，自动折回 `<think>` 记录 |
| token 计数 | 本地 HF checkpoint tokenizer | chars/4 近似（无需本地权重） |
| 聚合脚本 | 有两个 TypeError（多传参数），跑不通 | 已修复；输出改为 BCP evaluator 兼容格式 |
| docid 记录 | 无 | 每条轨迹记录 retrieved_docids/chunk_ids；聚合时对全部并行轨迹取并集（含从复用前缀的 tool_response 文本中回收的 docid） |

方法本体（分支点选择数学、采样预算核算、REPORT/INTEGRATE prompt、采样参数
0.6/0.95/1.1）与原版保持一致。

## 运行前提

1. `pip install openai numpy json5 tqdm pymilvus requests aiohttp`
2. retriever 配置复用 `inference/configs/bcp_baseline_retriever.json`（改好 `milvus.uri`）
3. `export OPENROUTER_API_KEY=...`

## 实验矩阵：ParallelMuse 和什么配合使用

ParallelMuse 自带完整的 ReAct 循环（`rollout_single_traj` 就是 Tongyi ReAct
推理循环的 async 复刻），**不需要**外接任何 agent 框架，只需要一个 base model。
论文对比条目建议成对出现，唯一变量是并行方法：

| 条目 | 命令 | 每题成本 |
| --- | --- | --- |
| model + ReAct（对照组） | Step 1 用 `--sampling_budget 1` → `bcp_traj_to_eval.py` 转评测格式 | 1 条轨迹 |
| model + ParallelMuse（实验组） | 完整三步流程 | N 条轨迹 + N 次报告 + 1 次聚合 |

两组共用同一 retriever、同一 prompt、同一采样参数与评测输出格式。

推理模型开关：如需 OpenRouter 的 reasoning 模式（与 deepsearch_edi 自家
qwen3 实验的 `extra_body` 用法对齐），两个脚本都支持
`--extra-body '{"reasoning": {"enabled": true}}'`。

冒烟测试：`--limit 1` 只跑第一题。

## 三步流程

```bash
cd WebAgent/ParallelMuse

# Step 1: 初始轨迹级并行 rollout（mode=none）
python bcp_partial_rollout.py \
  --qa_file_path /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
  --output_dir ./bcp_results \
  --llm-model deepseek/deepseek-v4-flash \
  --partial_sampling_mode none --sampling_budget 2 \
  --max_llm_workers 16 --max_search_workers 8

# Step 2: 不确定性引导的部分 rollout（读取 Step 1 的 initial_rollout 文件）
python bcp_partial_rollout.py \
  --qa_file_path /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
  --output_dir ./bcp_results \
  --llm-model deepseek/deepseek-v4-flash \
  --partial_sampling_mode tool_call_ppl \
  --initial_rollout_num 2 --partial_sampling_topk 2 \
  --partial_sampling_times_per_pos 1 --sampling_budget 8

# Step 3: 压缩报告 + 聚合，输出 BCP evaluator 格式
python bcp_aggregate.py \
  --rollout-file ./bcp_results/queries_deepseek_deepseek-v4-flash_2_tool_call_ppl_2_1_1.jsonl \
  --output-dir ./bcp_results/aggregated \
  --llm-model deepseek/deepseek-v4-flash --store-reports

# （对照组）ReAct 单轨迹基线：Step 1 改 --sampling_budget 1，然后转评测格式
python bcp_traj_to_eval.py \
  --rollout-file ./bcp_results/queries_deepseek_deepseek-v4-flash_1_none_initial_rollout.jsonl \
  --output-dir ./bcp_results/react_baseline_eval \
  --model-label deepseek/deepseek-v4-flash
```

## N 次独立 ReAct rollout（YAML 编排）

跑"同一配置 × 8 个独立环境"（如 gpt-oss-20b 的 8 次 ReAct，对齐
deepsearch_edi 的 `openai_gpt_oss_20b_react_baseline_512tok_verbatim_0521.json`）：

```bash
# 一次并发 3 个环境，跑完自动补位，直到 8 个全完成；可 Ctrl-C 后重跑续传
python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --max-parallel 3

# 查看各环境进度
python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --status

# 只跑指定环境
python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --only env1,env2

# 冒烟测试：把 YAML 里 common.limit 改为 1 再跑
```

每个环境独立目录：轨迹 jsonl + `orchestrator.log` + `env_meta.json`（记录该
环境的完整配置、起止时间、退出码、进度）+ `eval/`（`convert_to_eval: true`
时自动转出的 evaluator 文件）。环境完成的判定是"每题都有 ≥ sampling_budget
条轨迹"，所以中断重跑既不会重复也不会漏。

预算关系（原版约束，Step 2 会 assert）：
`sampling_budget >= initial_rollout_num * (1 + topk * rounds * times_per_pos)`。
上例 2 * (1 + 2*1*1) = 6 <= 8，多出的 2 条会补成普通轨迹级 rollout。

**简化路线**：如果 provider 不返回 logprobs（跑一条后看输出里 step_ppl 是否全为
null），PPL 分支的意义就没了——此时直接 `--partial_sampling_mode none
--sampling_budget N` 跑纯并行 + Step 3 聚合，这等价于论文的 aggregation-only
消融，同样是合法的 ParallelMuse 实验条目（论文 Table 里有对应设置）。

## 输出

Step 3 在 `--output-dir` 下每题一个 `run_*.json`：`query_id` / `status` /
`retrieved_docids`（全部并行轨迹的并集，docid 非 chunk_id）/
`result[0].output`（聚合后的最终答案）/ metadata（method、model、n_trajectories、
retriever 溯源）。可直接喂 BrowseComp-Plus evaluator。重跑会跳过已 completed
的 query_id。

## 竞品实验操作手册（gpt-oss-20b，BCP 前 100 条，预算 8）

按论文方法（每题 8 条 = 1 初始 + 2 分支点 × 3 续写 + 1 补充 → 压缩聚合出 1 个答案）：

```bash
cd WebAgent/ParallelMuse
export OPENROUTER_API_KEY=...
OUT=./bcp_results/pm_gpt_oss_20b
QA=/path/to/BrowseComp-Plus/topics-qrels/queries.tsv
M="openai/gpt-oss-20b"; EB='{"reasoning": {"enabled": true}}'

# 0. 冒烟 + logprobs 探测（1 题）
python bcp_partial_rollout.py --qa_file_path $QA --limit 1 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 1
python -c "import json;d=[json.loads(l) for l in open('$OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl')];print('step_ppl:',[m.get('step_ppl') for t in d for m in t['rollout'] if m['role']=='assistant'][:3])"

# 若 step_ppl 有数值 → 完整方法：
# 1. Step A 初始轨迹（每题 1 条）
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 1
# 2. Step B 不确定性引导部分 rollout，补到每题 8 条
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode tool_call_ppl \
  --initial_rollout_num 1 --partial_sampling_topk 2 \
  --partial_sampling_times_per_pos 3 --sampling_budget 8
# 3. Step C 聚合
python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_tool_call_ppl_2_1_3.jsonl \
  --output-dir $OUT/aggregated --llm-model $M --extra-body "$EB" --store-reports

# 若 step_ppl 全为 null → 纯并行路线（论文的 trajectory-level 并行 + 聚合设置）：
python bcp_partial_rollout.py --qa_file_path $QA --limit 100 --output_dir $OUT \
  --llm-model $M --extra-body "$EB" --partial_sampling_mode none --sampling_budget 8
python bcp_aggregate.py \
  --rollout-file $OUT/queries_openai_gpt-oss-20b_1_none_initial_rollout.jsonl \
  --output-dir $OUT/aggregated --llm-model $M --extra-body "$EB" --store-reports
```

`$OUT/aggregated/` 即 BrowseComp-Plus evaluator 的输入目录。论文需报告：
方法变体（uncertainty-guided vs trajectory-level）、每题轨迹数（metadata.
n_trajectories）、总 search 次数、retriever 溯源（metadata 自带）。

## 公平性口径

- 检索面与 Tongyi ReAct 竞品实验完全一致：同一 Milvus collection、同一 embedding
  与 instruction、hybrid WeightedRanker(0.6,0.4)、top_k=3、baseline 返回策略，
  参数在 retriever 配置里锁死，模型只能传 query 字符串。
- ParallelMuse 属于 test-time scaling：论文对比时应标注每题的轨迹数
  （metadata.n_trajectories）与总 search 次数，与单轨迹 ReAct 条目区分预算。
- `retrieved_docids` 取并集是诚实口径：所有检索都来自固定 retriever，聚合答案
  可能引用任何一条轨迹的证据。
