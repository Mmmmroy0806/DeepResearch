# Tongyi DeepResearch 接入 BrowseComp-Plus Baseline Retriever 方案

本文档是给后续执行 agent 的完整任务说明。执行 agent 不具备前序讨论上下文，因此本文档会同时包含背景、已经确定的设计取舍、排除项、推荐实现路径、验收标准和关键代码位置。

## 0. 给执行 agent 的背景说明

请先理解这个任务的实验目的，再开始改代码。

用户正在准备/投稿 `deepsearch_edi` 相关论文，需要跑竞品系统 Tongyi DeepResearch 作为对比实验。这里不是单纯“让 Tongyi 能跑 BrowseComp-Plus”，而是要保证论文实验中不同系统之间的变量被控制住。

论文中用户有两套 retriever：

- `baseline retriever`
- `LEGO retriever`

本任务只处理 `baseline retriever`。LEGO 是后续另一个检索器，不应在本任务中接入、引用、混合或预留不清晰的半成品逻辑。

这个实验想回答的问题不是：

```text
Tongyi DeepResearch 自己联网搜索 BrowseComp-Plus 能做到什么？
```

而是：

```text
在使用同一个 BrowseComp-Plus baseline retriever 的前提下，Tongyi DeepResearch 这个竞品 agent/LLM 的表现如何？
```

因此本任务的第一性原则是：

```text
固定 retriever，替换 agent/LLM。
```

换句话说，Tongyi DeepResearch 在实验里不是一个完整的“自带联网搜索系统”，而是一个接入固定检索器的 reasoning/search agent。它可以决定搜什么 query、何时继续搜、如何综合证据，但它不能决定用什么搜索引擎、什么索引、什么 top_k、是否联网、是否读取网页、是否 rerank。

如果执行过程中遇到不确定情况，请用下面的判断原则：

1. 会不会改变 retriever 结果？如果会，先不要做。
2. 会不会让 Tongyi 绕过 baseline retriever 去联网？如果会，必须禁止。
3. 会不会让不同系统使用不同检索配置？如果会，必须禁止。
4. 会不会影响 BrowseComp-Plus evaluator 需要的 `retrieved_docids`？如果会，必须显式修正。
5. 如果只是优化工程结构，但不改变检索结果和输出格式，可以做。

## 0.1 为什么不是“随便给 Tongyi 加个本地搜索”

这个任务最容易做错的地方，是把它理解成：

```text
把 Tongyi 的 Google Search 换成某种本地搜索即可。
```

这不够。论文实验要求的是“同一套 baseline retriever”，而不是“随便一个本地 retriever”。所以 adapter 必须尽量复刻 `deepsearch_edi` baseline 的行为，包括：

- query embedding 前的 instruction
- Milvus collection
- dense/sparse/hybrid 模式
- WeightedRanker 权重
- top_k
- baseline chunk 返回策略
- docid/chunk_id 处理

如果只是使用 BrowseComp-Plus 官方 BM25/FAISS，或者只用 Milvus sparse，或者让 Tongyi 自己调 Serper/Jina，都不能算“使用用户论文中的 baseline retriever”。

## 0.2 为什么要做 search-only

Tongyi 原始 DeepResearch inference 是一个联网 agent，默认工具包括：

- `search`
- `visit`
- `google_scholar`
- `parse_file`
- `PythonInterpreter`

但 BrowseComp-Plus 的设计是固定语料检索，目的是隔离 retriever 与 agent 的影响。如果保留 `visit` 或 Jina/Serper，Tongyi 就可能访问 live web 或额外网页内容，实验不再公平。

所以本任务要求 BCP 模式只暴露一个 `search` 工具。这不是为了简化实现，而是为了实验控制。

正确实验边界：

```text
Tongyi 可以多轮调用 search，但 search 的所有结果只能来自 fixed BrowseComp-Plus Milvus baseline index。
```

错误实验边界：

```text
Tongyi search 到 URL 后再 visit URL。
Tongyi search 失败后 fallback 到 Google/Serper。
Tongyi 用 Scholar 补充证据。
Tongyi 使用 Python 或文件工具读取额外语料。
```

## 0.3 为什么不需要单独部署 retriever service

前序讨论中曾考虑过单独部署一个 HTTP retriever service。那个方案在工程边界上很干净，但用户后来澄清：实际希望最终只运行 Tongyi DeepResearch 代码和 server 上的 Milvus。

因此本任务的推荐实现是：

```text
retriever 逻辑作为 Tongyi 进程内 Python adapter 存在。
```

也就是说：

- Milvus 是远端/server 上的索引服务。
- baseline retriever 是 Tongyi 代码里的 Python 查询逻辑。
- Tongyi 的 `search` tool 接收 query，然后内部计算 embedding 并查询 Milvus。

不需要额外启动：

```text
baseline-retriever-service
```

但要注意，不部署 retriever service 不等于“不需要 retriever 逻辑”。retriever 逻辑仍然必须存在，只是嵌在 Tongyi 的 search tool 里。

## 0.4 为什么不建议直接 import deepsearch_edi 整包

用户的 `deepsearch_edi` 是论文代码，当前正在投稿。竞品适配代码不应反向污染或破坏论文主代码。

此外两个项目依赖不同。直接让 Tongyi import 整个 `openjiuwen_deepsearch` 可能带来：

- Python path 问题
- openjiuwen 依赖冲突
- pymilvus 版本冲突
- 配置对象不兼容
- 后续别人复现实验时环境复杂

所以推荐从 `deepsearch_edi` 抽取最小 baseline retriever 逻辑，在 Tongyi `inference/` 下实现一个独立 adapter。这个 adapter 的行为要和 `deepsearch_edi` baseline 对齐，而不是为了“少写代码”去强行 import 整个项目。

如果执行 agent 最终决定直接 import `deepsearch_edi` 的具体类，也必须先证明：

- 不修改 `deepsearch_edi`
- 不破坏 Tongyi 原依赖
- 能稳定运行
- 同 query 与原 baseline 返回 top-k chunk id 一致

否则优先走最小复制实现。

## 0.5 这个方案要给论文留下什么证据

这不是临时脚本。最终实验产物要能支撑论文中的方法描述和可复现性。

因此代码和日志中最好能明确记录：

- 使用的 retriever 名称：`deepsearch_edi_baseline_milvus`
- 使用的 Milvus collection
- 使用的 mode，例如 `hybrid`
- `top_k`
- `baseline=true`
- `add_instruction`
- embedding model/base_url，不要记录明文 key
- 每次 search 的 query
- 每次 search 返回的 chunk ids
- 每题累计的 retrieved docids

这些信息不仅用于 debug，也用于证明 Tongyi 的竞品实验没有偷用自己的联网能力或不同检索配置。

## 1. 背景与目标

当前需要为论文实验跑竞品模型/agent：Alibaba Tongyi DeepResearch。

用户自己的项目在：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi
```

竞品 Tongyi DeepResearch 当前工作目录在：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch
```

论文里有两类 retriever：

- `baseline retriever`
- `LEGO retriever`

本任务只接入 `baseline retriever`。不要把 LEGO 逻辑混进来，也不要引入新的 reranker 或改动检索策略。

最终目标：

```text
让 Tongyi DeepResearch 在运行 BrowseComp-Plus 时，使用与 deepsearch_edi 论文 baseline 完全相同的 retriever/index/config。
```

这里的 “相同 retriever” 指：

- 同一个 BrowseComp-Plus Milvus collection
- 同一种 query embedding 配置
- 同样的 top_k、mode、baseline、chunk 策略
- 同样的 baseline 后处理逻辑
- Tongyi 不能走自己的联网搜索、Jina 读取网页、Serper/Google 搜索、Scholar、文件解析等工具

Tongyi DeepResearch 在这个实验中只负责：

- 生成 search query
- 读取 search 工具返回的 snippets
- 决定是否继续检索
- 输出最终答案

实际检索必须由 baseline retriever 完成。

## 2. 已经确认和排除的事情

### 2.1 不需要单独部署 retriever service

前序讨论中曾考虑过 `baseline-retriever-service`，但用户澄清最终运行环境是：

```text
Tongyi DeepResearch 代码 + server 上的 Milvus
```

因此不要求额外部署一个 retriever HTTP 服务。

推荐做法是：

```text
把 baseline retriever 作为 Tongyi 进程内的 Python adapter，挂到 Tongyi 的 search 工具里。
```

实际查询流：

```text
Tongyi 模型输出 search tool_call
  ↓
Tongyi 的 Search.call() 接收 query
  ↓
Search.call() 内部调用 BaselineMilvusRetriever
  ↓
BaselineMilvusRetriever 调 embedding API 生成 query vector
  ↓
BaselineMilvusRetriever 使用 pymilvus 查询远端/server Milvus
  ↓
Milvus 返回 top-k chunks
  ↓
Search.call() 格式化 snippets 给 Tongyi 模型
```

注意：Milvus 是服务；retriever 是 Python 查询逻辑。query 不是直接发给 Milvus，而是先进入 Tongyi 的 search tool，再由 search tool 计算 embedding 并查询 Milvus。

### 2.2 不要让 Tongyi 继续使用联网工具

BrowseComp-Plus 是固定语料检索 benchmark，不能让 Tongyi 使用联网搜索能力。

因此 BCP 模式下必须禁用：

- `visit`
- `Jina`
- `Serper`
- `Google Search`
- `Google Scholar`
- `parse_file`
- `PythonInterpreter`
- 任何外部网页访问或联网搜索

Tongyi 只能看到一个工具：`search`。

### 2.3 不建议直接 import 整个 deepsearch_edi

`deepsearch_edi` 依赖 `openjiuwen`、`pymilvus` 等；Tongyi 依赖 `qwen-agent` 等。直接跨项目 import 整包容易引起依赖/路径冲突。

推荐：

- 从 `deepsearch_edi` 抽取/复制 baseline retriever 所需的最小逻辑到 Tongyi `inference/` 下。
- 保持逻辑与配置一致。
- 不要改 deepsearch_edi 主体，避免影响论文投稿代码。

如果执行 agent 有信心管理依赖，也可以让 Tongyi adapter import deepsearch_edi 的具体类，但最稳妥的是复制最小实现。

### 2.4 baseline retriever 不是单独 class，而是同一个 Milvus retriever 的模式

`deepsearch_edi` 中 baseline 逻辑在：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/algorithm/search_tools/retrieval/retriever.py
```

核心类：

```python
BrowsecompPlusMilvusRetriever
```

baseline 模式由 `RetrieveConfig.baseline=True` 控制。它的行为是：

- `search_limit = top_k`
- 不使用 `top_k * top_k_multiply_factor` over-fetch
- 不按 docid 去重
- 不合并同文档 chunks
- 每个 hit 作为一个独立返回结果

这和非 baseline 模式不同。非 baseline 会 over-fetch、按 docid merge、可能补 chunk 0。不要把非 baseline 行为带进 Tongyi。

## 3. 关键代码位置

### 3.1 deepsearch_edi baseline retriever 参考代码

配置：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/config/config.py
```

重点类：

- `MilvusConfig`
- `RetrievalSettingsConfig`

检索工具封装：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/algorithm/search_tools/retriever_tool.py
```

重点类：

- `Retrieve`
- `RetrieveBrowsecompPlus`

retriever 主逻辑：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/algorithm/search_tools/retrieval/retriever.py
```

重点类/函数：

- `BrowsecompPlusMilvusRetriever.retrieve`
- `_hits_to_baseline_combined`
- `_return_unique`，仅作反例参考，baseline 不应调用

embedding：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/algorithm/search_tools/retrieval/embedder.py
```

重点类：

- `RemoteQwenEmbedder`

Milvus index 构建/schema：

```text
/Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi/openjiuwen_deepsearch/algorithm/search_index/create_browsecompplus_index.py
```

重点字段：

- `id`
- `docid`
- `embedding`
- `content`
- `content_sparse`
- `title`
- `metadata`

### 3.2 Tongyi DeepResearch 当前入口

原始 search tool：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/tool_search.py
```

原始 agent：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/react_agent.py
```

原始 prompt：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/prompt.py
```

原始 runner：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/run_multi_react.py
```

原始 `react_agent.py` 当前会加载：

```python
TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
```

BrowseComp-Plus 模式下不要使用这套工具列表。

### 3.3 可参考的 BrowseComp-Plus 官方 Tongyi 适配

本机还有一个 BrowseComp-Plus repo：

```text
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus
```

里面已有 Tongyi search-only 适配，可作为结构参考：

```text
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/search_agent/tongyi_utils/tool_search.py
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/search_agent/tongyi_utils/react_agent.py
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/search_agent/tongyi_client.py
```

这个 repo 里提到的 baseline retriever 有 BM25 和 FAISS 两类，但本任务不是接官方 BM25/FAISS，而是接 `deepsearch_edi` 的 baseline Milvus retriever。

## 4. 推荐新增文件

建议不要污染 Tongyi 原始联网版本，而是新增 BrowseComp-Plus 专用实现：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/
  baseline_retriever.py
  tool_search_baseline.py
  prompt_bcp.py
  react_agent_bcp.py
  run_browsecomp_plus.py
  configs/bcp_baseline_retriever.json
```

如果不想新增太多文件，也可以只新增：

```text
baseline_retriever.py
tool_search_baseline.py
react_agent_bcp.py
run_browsecomp_plus.py
```

但请确保 BCP 运行不再 import 原始 `tool_search.py`、`tool_visit.py`、`tool_scholar.py` 等联网工具。

## 5. 配置设计

建议新增：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/configs/bcp_baseline_retriever.json
```

示例：

```json
{
  "milvus": {
    "uri": "http://server-ip:19530",
    "token": "",
    "database_name": "deepsearch_benchmarks",
    "collection_name": "browsecompplus_v2_baseline_512tok"
  },
  "embedding": {
    "model_name": "qwen/qwen3-embedding-8b",
    "api_key": "${OPENROUTER_API_KEY}",
    "base_url": "https://openrouter.ai/api/v1/embeddings",
    "timeout": 100
  },
  "retrieval": {
    "top_k": 3,
    "top_k_multiply_factor": 5,
    "mode": "hybrid",
    "add_instruction": true,
    "baseline": true,
    "include_leading_chunk": false,
    "snippet_max_tokens": 512
  }
}
```

重要：

- `collection_name` 必须使用论文 baseline collection。
- 如果论文 baseline collection 不是 `browsecompplus_v2_baseline_512tok`，请改成论文实际使用的 collection。
- `top_k` 必须与论文 baseline 保持一致。
- `mode` 必须与论文 baseline 保持一致。如果论文 baseline 是 `hybrid`，Tongyi 也必须是 `hybrid`。
- `add_instruction` 必须一致。`RemoteQwenEmbedder` 中 query instruction 是：

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

- `baseline` 必须为 `true`。
- `include_leading_chunk` 对 baseline 通常不会生效，因为 baseline 不做 merge，但仍建议固定为 `false`，避免未来代码误用。

### 5.1 关于 Milvus 连接

`deepsearch_edi` 当前 `BaseRetriever` 中使用：

```python
MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
```

但实际 server 上的 Milvus 可能需要：

```python
MilvusClient(uri=milvus_uri, token=milvus_token)
```

请在 Tongyi 的 `baseline_retriever.py` 中直接支持 `uri` 和 `token`。

兼容建议：

- 配置里优先使用 `milvus.uri`
- 如果只给 `host/port`，再拼接为 `http://host:port`
- `token` 为空时不传或传空值均可，但要测试 pymilvus 行为

### 5.2 关于 embedding

如果检索模式是 `dense` 或 `hybrid`，必须有 query embedding。

仅有 Milvus 不够。query 需要先调用 embedding API 得到 vector，再查询 Milvus。

如果只用 `sparse`/BM25，理论上可以不需要 embedding。但本任务要求与论文 baseline 一致，不要擅自切成 sparse。

## 6. baseline_retriever.py 实现要求

目标：在 Tongyi 代码内实现一个最小可用的 `BaselineMilvusRetriever`。

建议接口：

```python
class BaselineMilvusRetriever:
    def __init__(self, config: dict):
        ...

    def search(self, queries: list[str]) -> list[dict]:
        ...
```

返回格式建议：

```python
[
    {
        "query": "...",
        "hits": [
            {
                "rank": 1,
                "chunk_id": "...",
                "docid": "...",
                "title": "...",
                "text": "...",
                "score": 0.0,
                "metadata": {...}
            }
        ]
    }
]
```

### 6.1 需要复制/实现的逻辑

从 `deepsearch_edi` 复制最小必要逻辑：

- `RemoteQwenEmbedder.encode`
- query instruction 逻辑
- Milvus `dense` search
- Milvus `sparse` search
- Milvus `hybrid_search`，使用：

```python
WeightedRanker(0.6, 0.4)
```

- `mode` 支持至少：
  - `dense`
  - `sparse`
  - `hybrid`

可以先不支持 `grep-*`，除非论文 baseline 需要。

### 6.2 baseline 后处理

baseline 模式下：

```python
search_limit = top_k
```

每个 Milvus hit 直接转成一个 hit 返回，不要做：

- docid merge
- chunk 0 backfill
- rerank
- 去重
- full document expansion

### 6.3 metadata 兼容

历史日志里出现过类似错误：

```text
BrowsecompPlusHitMetadata() argument after ** must be a mapping, not NoneType
```

因此实现时要防御 `metadata is None` 的情况。

当 Milvus hit 没有 metadata 时，至少要 fallback：

```python
title = hit.entity.get("title", "")
chunk_id = hit.id
docid = hit.entity.get("docid") or parse_docid_from_chunk_id(chunk_id)
text = hit.entity.get("content", "")
metadata = {}
```

chunk id 到 docid 的解析建议：

```python
def chunk_id_to_docid(chunk_id: str) -> str:
    if "__" in chunk_id:
        return chunk_id.rsplit("__", 1)[0]
    return chunk_id
```

BrowseComp-Plus evaluator 需要 docid，不是 chunk id。

## 7. tool_search_baseline.py 实现要求

目标：替换 Tongyi 的 `search` 工具，但工具名仍然叫 `search`。

建议：

```python
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Search the fixed BrowseComp-Plus corpus with the baseline retriever."
    parameters = ...
```

输入兼容两种形式：

```json
{"query": "single query"}
```

以及原 Tongyi 常见形式：

```json
{"query": ["query1", "query2"]}
```

但 prompt 中建议只要求单 query，减少模型一次性发很多 query 导致 context 爆炸。

返回给模型的文本格式建议保持类似 Tongyi 原格式：

```text
A search for '...' found 3 results:

## Search Results

1. [title]
docid: ...
chunk_id: ...
score: ...
snippet text...

2. [title]
...
```

同时 `Search.call()` 最好返回二元组：

```python
return response_text, docids, chunk_ids
```

如果为了兼容原始 `react_agent` 只接受字符串，可以在 BCP 专用 `react_agent_bcp.py` 中处理这个返回值。

## 8. prompt_bcp.py 实现要求

不要使用原始 `SYSTEM_PROMPT`，因为原 prompt 声明了 `visit`、`google_scholar`、`parse_file`、`PythonInterpreter` 等工具。

新 prompt 只声明 `search`：

```text
You are a deep research assistant ...

# Tools
You may call the search tool to search a fixed BrowseComp-Plus knowledge corpus.

<tools>
{"type": "function", "function": {"name": "search", "description": "Search the fixed BrowseComp-Plus corpus with the baseline retriever. It returns top relevant evidence snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}
</tools>
```

不要写：

- Google web search
- webpage
- URL
- visit page

因为本实验没有联网浏览。

## 9. react_agent_bcp.py 实现要求

可以参考原：

```text
/Users/mmmroym/Downloads/huawei/DeepResearch/inference/react_agent.py
```

也可以参考：

```text
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/search_agent/tongyi_utils/react_agent.py
```

BCP 版 agent 要求：

- 只 import `tool_search_baseline.Search`
- 只初始化 search tool
- 不 import 原始 `tool_visit.py`
- 不 import 原始 `tool_scholar.py`
- 不 import 原始 `tool_file.py`
- 不 import 原始 `tool_python.py`

记录字段：

```python
tool_call_counts = {}
tool_call_counts_all = {}
retrieved_docids = []
retrieved_chunk_ids = []
```

每次工具调用：

- `tool_call_counts_all["search"] += 1`
- 如果检索成功：
  - `tool_call_counts["search"] += 1`
  - 追加 docids/chunk_ids

最终输出里必须包含：

```json
{
  "question": "...",
  "answer": "...",
  "prediction": "...",
  "termination": "...",
  "messages": [...],
  "tool_call_counts": {"search": 8},
  "tool_call_counts_all": {"search": 8},
  "retrieved_docids": ["..."],
  "retrieved_chunk_ids": ["..."]
}
```

## 10. run_browsecomp_plus.py 实现要求

输入格式建议支持：

1. BrowseComp-Plus 的 `topics-qrels/queries.tsv`

```text
query_id<TAB>query
```

2. JSONL：

```json
{"query_id": "...", "question": "...", "answer": "..."}
```

输出建议每题一个 JSON，方便 BrowseComp-Plus evaluator：

```json
{
  "metadata": {
    "model": "...",
    "retriever": "deepsearch_edi_baseline_milvus",
    "collection_name": "...",
    "mode": "hybrid",
    "top_k": 3,
    "baseline": true,
    "add_instruction": true
  },
  "query_id": "...",
  "question": "...",
  "status": "completed",
  "tool_call_counts": {"search": 8},
  "tool_call_counts_all": {"search": 8},
  "retrieved_docids": ["..."],
  "retrieved_chunk_ids": ["..."],
  "result": [
    {
      "type": "output_text",
      "output": "final answer"
    }
  ],
  "raw_messages": [...]
}
```

BrowseComp-Plus evaluator 至少需要：

- `query_id`
- `status`
- `retrieved_docids`
- `result` 中的 final output

可参考官方 repo 的输出格式：

```text
/Users/mmmroym/Downloads/huawei/gnosis_jiuwen/BrowseComp-Plus/search_agent/tongyi_client.py
```

## 11. 依赖

Tongyi `requirements.txt` 当前未必包含：

```text
pymilvus
```

需要添加或在运行环境安装。

建议先检查：

```bash
python -c "import pymilvus; print(pymilvus.__version__)"
```

如果没有，则需要安装与 server Milvus 兼容的版本。`deepsearch_edi` 的 lock 中出现过 `pymilvus 2.6.9`，但最终以当前 Milvus server 兼容性为准。

不要引入 `openjiuwen` 作为 Tongyi 必需依赖，除非确实决定直接复用 deepsearch_edi 包。最小实现只需要：

- `pymilvus`
- `requests`
- `pydantic` 可选
- `transformers` 仅用于 snippet token truncation，可选

## 12. 测试与验收标准

### 12.1 单元级测试

先写一个最小脚本或在 `baseline_retriever.py` 加 `__main__`：

```bash
python inference/baseline_retriever.py \
  --config inference/configs/bcp_baseline_retriever.json \
  --query "Aloinopsis Washingtonia difference"
```

验收：

- 能连上 server Milvus
- 能调用 embedding API
- 能返回 top-k hits
- 每个 hit 有：
  - `docid`
  - `chunk_id`
  - `text`
  - `score`

### 12.2 与 deepsearch_edi 对齐测试

必须选 3-5 个 query，对比：

1. `deepsearch_edi` 原 baseline retriever 返回的 top-k chunk ids
2. Tongyi adapter 中 `BaselineMilvusRetriever` 返回的 top-k chunk ids

验收：

```text
同 query、同 config、同 collection 下，top-k chunk_id 完全一致。
```

如果不一致，优先检查：

- collection 是否一致
- embedding model/base_url/api_key 是否一致
- query instruction 是否一致
- `mode` 是否一致
- WeightedRanker 权重是否一致
- top_k 是否一致
- 是否误做了 doc merge 或去重

### 12.3 Tongyi agent 工具隔离测试

跑 1 条 query：

```bash
python inference/run_browsecomp_plus.py ...
```

验收日志中不能出现：

- `Jina`
- `Serper`
- `Google search`
- `visit`
- `google_scholar`
- `parse_file`
- `PythonInterpreter`

只能出现 `search`。

### 12.4 输出格式测试

检查输出 JSON：

- `retrieved_docids` 非空
- `retrieved_chunk_ids` 非空
- `retrieved_docids` 不是 `docid__chunk` 格式
- `result` 中有 final answer
- `status` 成功时为 `completed`

### 12.5 小批量测试

先跑 5-10 条 BrowseComp-Plus query：

- 检查是否有异常中断
- 检查 context 是否过长
- 检查 tool call 次数是否合理
- 检查返回 snippets 是否过长

如果 context 太长：

- 降低 `snippet_max_tokens`
- 降低 `top_k`
- 不要启用 full document
- 不要合并 chunks

不要通过开启 visit 或联网搜索解决。

## 13. 常见坑

### 13.1 把 chunk id 当成 docid

Milvus 主键通常是：

```text
{docid}__{chunk_idx}
```

BrowseComp-Plus evaluator 需要 docid。

必须同时记录：

```text
chunk_id = "{docid}__{chunk_idx}"
docid = "{docid}"
```

### 13.2 metadata 为 None

不要假设 `metadata` 一定存在。fallback 必须健壮。

### 13.3 误用了原始 Tongyi prompt

原始 prompt 会鼓励模型调用 visit 和网页工具。BCP 模式必须使用 search-only prompt。

### 13.4 误用了 Serper/Jina

如果 BCP 跑出来需要 `SERPER_KEY_ID` 或 `JINA_API_KEYS`，说明实现错了。

### 13.5 每次 search 都重新初始化 Milvus

不要在每次 `Search.call()` 中重新创建 retriever/MilvusClient。

应该：

- `Search.__init__` 初始化一次
- 后续所有 `call()` 复用同一个 retriever 实例

### 13.6 参数由模型传入

不要让模型通过 tool arguments 控制：

- `top_k`
- `mode`
- `baseline`
- `collection_name`
- `add_instruction`

模型只允许传 query。检索参数从配置读取并锁死。

## 14. 建议执行顺序

1. 在 Tongyi `inference/` 下新增 `configs/bcp_baseline_retriever.json`。
2. 新增 `baseline_retriever.py`，实现 embedding + Milvus baseline search。
3. 用单 query 测试 `baseline_retriever.py` 能返回 hits。
4. 与 `deepsearch_edi` 同 query 返回结果对齐，确认 top-k chunk_id 一致。
5. 新增 `tool_search_baseline.py`，注册 `search`，返回 formatted snippets。
6. 新增 `prompt_bcp.py`，只声明 search。
7. 新增 `react_agent_bcp.py`，只加载 baseline search tool，并记录 docids/chunk_ids。
8. 新增 `run_browsecomp_plus.py`，支持 `queries.tsv` 或 JSONL。
9. 跑 1 条 query，检查没有联网工具调用。
10. 跑 5-10 条 query，检查输出格式和稳定性。
11. 跑完整 BrowseComp-Plus。
12. 用 BrowseComp-Plus evaluator 评测。

## 15. 最终期望说明

完成后，论文实验可以描述为：

```text
For fair comparison, Tongyi DeepResearch is evaluated with the same BrowseComp-Plus baseline retriever used in our system. We replace Tongyi's original web search and browsing tools with a search-only tool backed by our fixed Milvus baseline retriever. The retriever uses the same collection, embedding model, query instruction, retrieval mode, top-k, and baseline chunk-return policy. No web browsing, Jina reading, Serper/Google search, or additional reranking is enabled.
```

中文表述：

```text
为保证公平比较，我们将 Tongyi DeepResearch 的原生联网搜索与网页浏览工具替换为仅包含 search 的本地检索工具。该 search 工具复用我方论文 baseline retriever 的同一 Milvus 索引、embedding 配置、检索模式、top-k 与 baseline chunk 返回策略。实验过程中不启用 Jina、Serper/Google、网页访问、Scholar、文件解析或额外 reranking。
```
