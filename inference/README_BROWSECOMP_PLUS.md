# Tongyi DeepResearch × BrowseComp-Plus（deepsearch_edi baseline retriever）

竞品对比实验适配：固定 retriever，替换 agent/LLM。Tongyi 在此模式下只暴露一个
`search` 工具，所有检索结果来自 deepsearch_edi 论文 baseline 的同一 Milvus 索引
（同 collection / embedding / query instruction / hybrid 模式 / top_k /
baseline chunk 返回策略）。不启用 visit / Serper / Jina / Google /
Scholar / parse_file / PythonInterpreter。

## 文件

| 文件 | 作用 |
| --- | --- |
| `configs/bcp_baseline_retriever.json` | retriever 配置（检索参数在此锁死，模型不可控制） |
| `baseline_retriever.py` | deepsearch_edi baseline Milvus retriever 的最小移植（仅依赖 pymilvus + requests） |
| `tool_search_baseline.py` | search-only 工具，`call()` 返回 `(text, docids, chunk_ids)` |
| `prompt_bcp.py` | 只声明 search 的系统 prompt |
| `react_agent_bcp.py` | BCP 版 ReAct agent，记录 tool_call_counts / retrieved_docids / retrieved_chunk_ids |
| `run_browsecomp_plus.py` | 批量 runner，输出 BrowseComp-Plus evaluator 兼容 JSON |
| `test_bcp_alignment.py` | 与 deepsearch_edi 原 retriever 的 top-k chunk_id 对齐测试 |

## 运行前提

1. `pip install pymilvus==2.6.9`（已加入 requirements.txt）
2. 修改 `configs/bcp_baseline_retriever.json` 中 `milvus.uri`（/`token`）指向 server Milvus；
   collection 必须是论文 baseline collection（默认 `browsecompplus_v2_baseline_512tok`）。
3. `export OPENROUTER_API_KEY=...`（配置中 `${OPENROUTER_API_KEY}` 会从环境变量展开）
4. 本地 vLLM/sglang 起好 Tongyi 模型服务（`--port` 对应）。

## 步骤

```bash
# 1. 单查询冒烟测试（只测 retriever，不需要 LLM 服务）
python inference/baseline_retriever.py \
  --config inference/configs/bcp_baseline_retriever.json \
  --query "Aloinopsis Washingtonia difference"

# 2. 与 deepsearch_edi 对齐测试（在 deepsearch_edi 的运行环境中执行）
cd /Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi
.venv/bin/python /Users/mmmroym/Downloads/huawei/DeepResearch/inference/test_bcp_alignment.py
# 期望输出：ALIGNMENT OK: all top-k chunk_ids identical.

# 3. 先跑 1 条，检查日志只出现 search、无 Serper/Jina/visit
python inference/run_browsecomp_plus.py \
  --dataset /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
  --model /path/to/Tongyi-DeepResearch-30B-A3B \
  --output-dir runs/tongyi_bcp_baseline --port 6001 --limit 1

# 4. 小批量（5-10 条）→ 全量（去掉 --limit，可加 --num-threads）
python inference/run_browsecomp_plus.py \
  --dataset /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
  --model /path/to/Tongyi-DeepResearch-30B-A3B \
  --output-dir runs/tongyi_bcp_baseline --port 6001 --num-threads 8
```

输出目录中每题一个 `run_*.json`，包含 `query_id` / `status` /
`retrieved_docids`（docid，非 chunk_id）/ `retrieved_chunk_ids` /
`result`（含 `output_text` 最终答案）以及 metadata 中的 retriever 溯源信息
（retriever 名称、collection、mode、top_k、baseline、add_instruction、
embedding model/base_url，不含 key）。断点续跑：重复执行会自动跳过已成功的 query_id。

## 注意

- `retrieval` 配置段必须保持 `baseline: true`、`mode: hybrid`、`top_k: 3`、
  `add_instruction: true`，与论文 baseline 一致；改动即破坏公平比较。
- `snippet_max_tokens: 0` 表示不截断——与 deepsearch_edi baseline 一致
  （该 collection 的 chunk 本身即 512 token）。
- 如果运行日志出现 `SERPER_KEY_ID` / `JINA_API_KEYS` 相关报错，说明误用了原始
  `react_agent.py`/`tool_search.py`；BCP 模式只能经 `run_browsecomp_plus.py` 入口。
