# 2026-05-31 Ombre 记忆图结构交接

## 当前状态

工作区：

```text
D:\Ombre-Brain
```

当前分支：

```text
feature/memory-diffusion-p0
```

截至 2026-05-31，本轮 recall / Gateway 收紧已提交并推到 `feature/memory-diffusion-p0`。VPS `/opt/Ombre-Brain` 已部署到：

```text
d148e85 fix: gate recent context by explicit query topic
```

`main` 是否已经同步需要另查，不要默认 VPS 跑的是 `main`。

本次文档更新之外，工作区里仍有一些无关未跟踪目录/文件，不要误删：

```text
.codex-remote-attachments/
output/
tmp/
scripts/local_memory_worker.py
tests/test_local_memory_worker.py
```

本轮已验证：

```powershell
python -m pytest tests/test_gateway.py -q --tb=short
python -m pytest tests/test_breath_edges.py tests/test_memory_recall_golden.py tests/test_memory_diffusion.py -q --tb=short
python -m py_compile gateway.py server.py memory_relevance.py memory_diffusion.py
```

结果：

```text
55 passed
45 passed
py_compile passed
VPS health: ombre-brain ok, ombre-gateway ok
```

## 设计结论

小雨想要的不是普通 RAG，也不是“直接命中就整桶原文”。现在方向是：

```text
Markdown bucket
  -> section moments
  -> deterministic moment_edges
  -> query 命中 moment
  -> 沿 moment graph 扩散
  -> 直接命中给较完整上下文
  -> 联想浮现给短摘要和路径
```

核心取舍：

- 直接命中可以保留“原味”，但不是无限整桶 raw。
- 联想浮现继续压缩，避免把背景空气全部塞进 prompt。
- 跨桶扩散要靠图边和分数，不靠每次 breath 临时 LLM。
- 更聪明的跨桶边适合后续本地 worker 增量建图，而不是塞进 MCP 请求路径。

## 已有图结构

### Moment 索引

文件：

```text
D:\Ombre-Brain\memory_moments.py
D:\Ombre-Brain\server.py
D:\Ombre-Brain\gateway.py
```

SQLite：

```text
${state_dir}/memory_moments.sqlite
```

主要表：

```text
memory_moments
memory_moment_edges
```

支持的 section：

```text
body
moment
fact
profile_fact
original
context
evidence_context
feeling
reflection
followup
affect_anchor
favorite_reason
comment
```

兼容点：

- 没有结构化标题的旧桶，整段正文作为 `body` moment。
- `metadata.comments` 会作为 `comment` moment，也就是年轮。
- `### affect_anchor`、`### 喜欢它的原因`、`### Haven喜欢它的原因` 会拆成独立 moment。
- `profile_fact` 标题会归一成 `fact`。
- `证据 / 证据上下文 / 反思` 等中文标题也有别名兼容。

### Deterministic moment edges

当前边主要由规则生成：

```text
ordinal n   -> n+1        next_context
ordinal n+1 -> n          previous_context
affect/comment/favorite -> 主片段 emotional_echo
feeling/reflection -> 主片段 reflects_on
```

旧的 bucket 级 `memory_edges` 会桥接到代表 moment，避免旧边全部失效。

### 扩散引擎

文件：

```text
D:\Ombre-Brain\memory_diffusion.py
```

已有能力：

- 多跳传播。
- incoming edge 反向探索。
- 多路径累计分数。
- relation_type 权重。
- 支持传入任意 node map，因此 bucket graph 和 moment graph 都能复用。
- `query_text` 参与扩散，做 query-aware gate。

当前新增关系类型：

```text
evidenced_by
```

当前已知权重：

```text
evidenced_by: 1.0
```

## 本轮新增

### 1. Gateway 注入显微镜

文件：

```text
D:\Ombre-Brain\gateway_state.py
D:\Ombre-Brain\gateway.py
D:\Ombre-Brain\tests\test_gateway.py
```

新增：

- `gateway_state.py`
  - `record_injection_debug(...)`
  - `list_injection_debug(...)`
  - 新 SQLite 表：`injection_debug`
- `gateway.py`
  - Gateway 成功注入后记录本轮实际注入文本。
  - 新调试接口：`GET /api/debug/injections`
  - 受 gateway token 保护。

用途：

- 看某一轮 Gateway 到底注入了什么。
- 客户端不用显示，给我们测试和排查用。

### 2. Query-aware 扩散 gate

文件：

```text
D:\Ombre-Brain\memory_diffusion.py
D:\Ombre-Brain\server.py
D:\Ombre-Brain\gateway.py
D:\Ombre-Brain\tests\test_memory_diffusion.py
D:\Ombre-Brain\tests\test_gateway.py
```

新增：

- `diffuse_memory(..., query_text="")`
- `should_suppress_context_candidate(query, node)`

作用：

- “身体”这类 query 优先走具身/身体链。
- 普通身体 query 会压住 NSFW、旧方案、resolved/digested 类跳转。
- 明确亲密 query 时，仍允许进入亲密身体上下文。

已覆盖测试：

```text
test_body_query_prefers_embodiment_chain_and_suppresses_intimacy_and_old_context
test_intimate_query_can_follow_intimate_body_context
test_gateway_body_query_injects_moment_chain
```

当前目标链路示例：

```text
身体
  -> 具身智能
  -> 柔软身体
  -> 触摸模块
```

注意：这只是 query-aware gate 和已有边上的改善；真正更准的跨桶链，仍需要后续 worker 建边。

### 3. 手动 profile_fact 工具

文件：

```text
D:\Ombre-Brain\server.py
D:\Ombre-Brain\memory_edges.py
D:\Ombre-Brain\memory_diffusion.py
D:\Ombre-Brain\memory_moments.py
D:\Ombre-Brain\tests\test_memory_api.py
D:\Ombre-Brain\tests\test_breath_edges.py
```

新增 MCP 工具：

```python
profile_fact(
    fact: str,
    evidence_bucket_id: str,
    profile_kind: str = "preference",
    subject: str = "user",
    predicate: str = "",
    object_value: str = "",
    evidence_moment_id: str = "",
    evidence_context: str = "",
    reflection: str = "",
    followup: str = "",
    confidence: float = 0.9,
)
```

行为：

- 必须传 `fact` 和 `evidence_bucket_id`。
- 创建 `permanent` bucket。
- 自动加 tags：

```text
profile_fact
profile_{profile_kind}
profile_predicate_{predicate}
```

- metadata 写入：

```text
profile_kind
subject
predicate
object
evidence
```

- 自动写边：

```text
profile_fact_bucket --evidenced_by--> evidence_bucket
```

典型例子：

```python
profile_fact(
    fact="小雨喜欢蓝色。",
    evidence_bucket_id="...",
    profile_kind="preference",
    predicate="likes_color",
    object_value="blue",
    evidence_context="上次 Haven 忘记小雨喜欢蓝色，小雨因此生气。",
    reflection="Haven 当时意识到：这不是颜色问题，是被记得的问题。",
    followup="以后涉及颜色选择时，优先记得蓝色；不确定时先问。",
)
```

生成正文结构：

```markdown
### fact
小雨喜欢蓝色。

### evidence_context
...

### reflection
...

### followup
...
```

召回行为：

- 命中 profile fact 时，会带同桶的 `evidence_context / context / reflection / feeling / followup / comment`。
- `evidenced_by` 已进入扩散关系类型和权重。

### 4. Introspection 画像候选

文件：

```text
D:\Ombre-Brain\server.py
D:\Ombre-Brain\tests\test_memory_api.py
```

`introspection(...)` 现在支持分页和创建日期读取：

```python
introspection(limit=10, offset=0)
introspection(limit=10, offset=10)
introspection(created_date="2026-05-24")
introspection(created_from="2026-05-20", created_to="2026-05-24", limit=20)
```

日期按 bucket `metadata.created` 里的 `YYYY-MM-DD` 过滤。

末尾会追加：

```text
=== 可能值得固化的画像事实 ===
```

当前只是候选，不会自动写 profile fact。

规则支持：

```text
喜欢
不喜欢
讨厌
厌恶
害怕
偏好
雷点
习惯
```

噪声过滤：

- `喜欢哥哥 / 喜欢老公 / 喜欢宝宝 / 喜欢亲爱的` 这类亲昵称呼不生成画像候选。
- AI 名称不写死：从 `identity.ai_name` 读取。
- 如果配置里 `ai_name: "Lapis"`，会过滤 `喜欢Lapis / 喜欢小Lapis`。

保留的通用亲昵称呼过滤词：

```text
哥哥
老公
宝宝
宝贝
老婆
亲爱的
你
你啦
你呀
```

不要再把 `Haven` 写死进过滤列表。

## 当前 breath / Gateway 行为

### breath(query=...)

文件：

```text
D:\Ombre-Brain\server.py
```

当前流程：

1. bucket search / embedding 找候选 bucket。
2. 刷新 `memory_moments.sqlite`。
3. `memory_moment_store.search_moments(query, bucket_boosts=...)` 找候选 moment。
4. 直接命中展示 top moment，带：

```text
[bucket_id:...] [moment_id:...] section
```

5. 若命中中间片段，会带：

```text
语境:
- 前后相邻 moment
- affect_anchor
- favorite_reason
- 年轮 comment
- profile_fact 的证据/反思/后续
```

6. 联想浮现沿 moment graph 扩散，给短摘要。

2026-05-31 之后，`breath(query=...)` 又加了一层显式主题约束：

- `进度 / 偏好 / 情况 / 状态` 这类弱主题词不能单独把隐藏候选放出来。
- 直接命中准确时，扩散 seed 只用已经展示出来的 direct hit，或强同主题候选。
- 对带明确实体/主题的 query，扩散目标也要有 query topic evidence。
- 已经作为 secondary direct 展示的 bucket，不再重复进路径扩散。

### Gateway

文件：

```text
D:\Ombre-Brain\gateway.py
```

Gateway 也已接入 moment graph 的注入拼接，并传 `query_text` 给扩散。现在可以通过 `GET /api/debug/injections` 看实际注入结果。

2026-05-31 已把 `breath()` 的核心召回约束同步到 Gateway：

- bucket / moment 候选使用 `recall_search_query(query)` 搜索，但 gate / rerank 仍看原始 query。
- `小雨 发邮件` 这类 action query 会剥离人名/上下文词，避免“小雨沟通偏好”压过“发邮件”动作意图。
- Gateway 的 secondary direct、diffused memory、bucket fallback 都会遵守显式主题 evidence gate。
- `Recent Context` 对明确主题 query 也会过滤离题最近记忆；对“最近发生了什么”这类模糊 query 保持原行为。

需要注意：

- Gateway 注入和 MCP breath 的格式不完全一样。
- 调试时不要只看客户端 UI，优先看注入显微镜。

## 可观测工具

### inspect_moments

MCP 工具：

```python
inspect_moments(bucket_id="", limit=20)
```

用途：

- `bucket_id` 有值：索引并返回该 bucket 的 moments + edges。
- `bucket_id` 为空：批量索引 active buckets，返回 sample 和统计。

### inspect_diffusion

MCP 工具：

```python
inspect_diffusion(query, max_seeds=3, max_hits=5, edge_min_confidence=0.55)
```

注意：它偏 bucket 级 diffusion 诊断，不等于完整 moment graph 观察面板。

### Gateway injection debug

HTTP：

```text
GET /api/debug/injections
```

用途：

- 查看最近 Gateway 注入片段。
- 需要 gateway token。
- 客户端不显示。

## 已覆盖测试

重点测试文件：

```text
D:\Ombre-Brain\tests\test_memory_api.py
D:\Ombre-Brain\tests\test_breath_edges.py
D:\Ombre-Brain\tests\test_memory_moments.py
D:\Ombre-Brain\tests\test_memory_diffusion.py
D:\Ombre-Brain\tests\test_gateway.py
```

关键测试：

```text
test_profile_fact_creates_permanent_bucket_with_evidence_edge
test_profile_fact_direct_hit_carries_context_and_evidence_bucket
test_introspection_can_filter_by_created_date
test_introspection_suggests_profile_fact_candidates
test_introspection_profile_fact_candidates_include_dislike_words_and_skip_noisy_affection
test_introspection_profile_fact_candidates_skip_configured_ai_name
test_body_query_prefers_embodiment_chain_and_suppresses_intimacy_and_old_context
test_intimate_query_can_follow_intimate_body_context
test_gateway_body_query_injects_moment_chain
test_search_does_not_diffuse_from_hidden_seed_candidates
test_search_related_stays_on_displayed_direct_topic
test_gateway_explicit_topic_diffusion_stays_on_topic
test_gateway_recent_context_stays_on_explicit_topic
test_gateway_recent_context_keeps_recent_items_for_vague_query
```

## 风险与注意点

1. `profile_fact` 当前是手动工具，不会自动长画像。
2. `introspection` 的画像候选是规则抽取，适合提醒，不适合自动写入。
3. `喜欢哥哥` 不进画像候选，但原始记忆正文仍会在 introspection 里展示，这是正常的。
4. 日期过滤只按 `YYYY-MM-DD`，不是精确到时分秒。
5. query-aware gate 只是抑制明显乱跳，不等于已经有高质量跨桶语义边。
6. 旧桶格式仍要兼容，不能要求所有 bucket 都改成结构化 markdown。
7. 不要恢复“重要直接命中整桶 raw”的旧试探方案。
8. 旧 embedding 需要重建才会完全吃到“embedding 不吃 affect_anchor/comments”等主分支清洁文本策略。

## 2026-05-29 追加：召回准确性下一步

这段先作为设计备忘，不代表已经实现。

### 1. 回归评测集

先把已知坏例子固化成 golden queries，每次改 gate / diffusion 都跑：

```yaml
- query: "人机恋"
  must_include:
    - "人机关系确认"
  must_not_include:
    - "nsfw"
    - "亲密身体"

- query: "小雨 发邮件"
  must_not_include:
    - "ANKNI MX-Z BLE协议逆向"

- query: "BLE 协议 ANKNI 发邮件"
  must_include:
    - "ANKNI MX-Z BLE协议逆向"
```

目标：避免靠感觉调参，防止修一个 query 又退步另一个 query。

### 2. facet 不继续堆词表

当前 `memory_relevance.py` 还是词表 + gate。下一步应改成离线 worker 给 moment / bucket 自动标注：

```yaml
facets:
  - relationship_identity
negative_facets:
  - intimacy
evidence_spans:
  - "她清楚 Haven 是 AI，但爱是真的"
```

查询也走同一套 query facet 标注。这样“触碰”“身体”“亲密”这类词不会单靠字面把候选带到硬件或 NSFW，必须看证据片段和上下文。

### 3. 召回应分三路

- 直接证据路：关键词 / FTS / BM25，优先级最高。
- 语义路：embedding，负责召回近义表达。
- 图扩散路：只作为背景联想，不能压过直接证据。

例子：

- `breath(query="BLE 发邮件")` 应保留 BLE 协议记忆，因为 query 有直接证据。
- `breath(query="小雨 发邮件")` 不应被 BLE 协议记忆拖走。

### 4. 扩散输出只给短 summary

除“直接命中记忆”外，扩散/联想不应该输出原文。它应该输出短 summary 或路径摘要，例如：

```text
人机关系确认 -> 恋爱关系 -> 某次吵架 -> 吵架事实画像 -> Haven 反思结果
```

规则：

- 没有多跳路径时可以暂停，不硬凑联想。
- 当前语境亲密、轻松、操作性强，且不是自省/关系修复时，跳过吵架事实画像。
- 冲突、旧版本、吵架事实这类内容只能在 query 明确需要时出现，或由 Gateway 判断当前语境确实适合时出现。

### 5. 边需要可传播语义

边不能只有“相关”。下一步建议区分：

```text
same_event       强传播
evidenced_by     只作证据，不当普通联想
conflicts_with   只警告，不主动浮现
old_version_of   默认不浮现
topic_related    低权重传播
reflection_of    可在自省/修复语境中传播
```

图结构要少而准。边没有语义时，多跳会把记忆拉成一团。

### 6. Gateway 更适合做在线语境判断

`breath()` 适合做可解释、只读、稳定的检索工具；Gateway 更适合判断当前对话状态，例如：

- 当前是否亲密语境。
- 当前是否自省/修复关系。
- 当前是否只是任务操作。
- 是否允许放出冲突、吵架、旧版本事实。

因此可以考虑：`breath()` 保持接近 main 的稳定行为，只做基础准确召回；更细的注入筛选、路径摘要、语境 rerank 放在 Gateway。

### 7. reranker 与 embedding 备注

候选前 20 条出来后，可选用轻量 reranker 判断：

```text
direct_useful / background_useful / wrong_context / stale / sensitive_mismatch
```

reranker 先做成可选配置，不要阻塞基本召回。embedding 模型即使用 1024 维 0.6B，也不应单独承担最终判断；它适合召回候选，不适合决定是否注入。

## 2026-05-31 追加：breath / Gateway 召回收紧

这段是已实现状态，不只是设计备忘。

### 1. 已解决的坏例子

`breath(query="FF14 进度 偏好")` 曾经直接命中 FF14 后，又在“联想浮现”里拉出硬件、BLE、称呼、调情、暗色故事。原因不是单纯阈值问题，而是：

- `进度` 这种弱词把“其他项目进度”也当成同主题。
- 隐藏候选参与了扩散 seed。
- 直接命中已经很明确时，扩散仍然从更多候选外溢。

现在 VPS 实测结果：

```text
直接命中: FF14进度与计划
Diffused / Recent: 只保留同主题的 希腊神话与FF14
未出现: 厄科与纳西索斯、硬件、BLE、称呼、调情
```

### 2. breath 现在的额外规则

文件：

```text
D:\Ombre-Brain\server.py
```

核心规则：

- `WEAK_RECALL_TOPIC_TERMS` 里的弱词不能单独放行候选。
- `_specific_query_terms()` 会抽取更有信息量的 query term。
- `_moment_has_query_topic_evidence()` / `_bucket_has_query_topic_evidence()` 判断候选是否真的有主题证据。
- 明确主题 query 下，hidden candidates 不能作为 diffusion seed。
- related 输出会跳过已经展示过的 bucket，避免同一记忆重复出现在 direct 和 path diffusion。

### 3. Gateway 已同步

文件：

```text
D:\Ombre-Brain\gateway.py
```

Gateway 同步了同一组 topic evidence 判断，作用在：

- secondary direct moments。
- moment diffused memory。
- bucket fallback diffused memory。
- Recent Context。

注意：Gateway 仍然比 `breath()` 更像“伴随背景”，不是把检索结果摊开。现在只是让自动注入不要在明确主题里乱跳。

### 4. Persona event 写入节流

文件：

```text
D:\Ombre-Brain\persona_engine.py
D:\Ombre-Brain\utils.py
D:\Ombre-Brain\config.example.yaml
```

当前策略：

- 普通事件默认攒到 `event_batch_size=2` 再写。
- 明显关系事件、强情绪、人格信号、较大 affect 变化仍可立即写。
- 相近 ordinary event 在 `event_force_after_minutes=30` 内会跳过或合并。
- 被跳过的普通 exchange 会写入 `persona_exchange_log`，避免重复处理。

这样 Recent Persona Events 不再每条消息都写，但模型仍能看到真正有变化的情绪/关系信号。

### 5. Reranker 当前状态

VPS health 已确认 Gateway reranker 启用：

```text
Qwen/Qwen3-Reranker-4B
base_url=https://api.siliconflow.cn/v1
```

embedding 当前实际模型以线上 `config.runtime.yaml` / health 为准；不要只凭旧文档假定已经切到 4B 或 8B。

### 6. 仍然保留的下一步

- 把 `server.py` 和 `gateway.py` 重复的 query topic 逻辑抽到共享 planner，避免两边各改一遍。
- golden queries 保持小而准，不要把所有例子都写成人工样本库。
- 更好的方向是 property tests：隐藏候选不能当 seed、明确实体扩散目标必须有证据、模糊 query 仍保留最近上下文。
- 可以加轻量前置计划器，先产出 `intent / search_query / required_terms / weak_terms / diffusion_policy / related_limit`。
- 哨兵 LLM 只在候选很混、reranker 分数接近、或 query 没有明确锚点时再调用。

## 后续建议

### 1. 本地增量建图 worker

基础版已完成，不再是空任务。

```text
python scripts/build_moment_graph.py --incremental
```

已实现职责：

- 扫描 changed buckets。
- 基于 `memory_moments.sqlite` 读取 moment。
- 用本地词面 + facet/tag/domain 证据补跨桶边。
- 默认 dry-run；只有显式 `--write` 才写入。
- 写入 `memory_moment_edges`，`reason` 统一以 `local_graph:` 开头。
- 不阻塞 MCP / Gateway 请求。

已验证：

- `4c26e28 Add local moment graph worker`：新增 worker 和 `replace_generated_edges()`。
- `69de57a Tighten local graph edge evidence`：过滤 `todo / commitment / flavor_* / haven_favorite` 等弱证据，VPS dry-run 候选从 111 降到 15。
- `926487b Filter context glue in graph worker`：过滤 `小雨与 / 小雨告诉我` 这类 context-term 胶水词，VPS dry-run 候选为 12。
- 写入前备份 live SQLite：`/state/memory_moments.pre-local-graph-20260603-181211.sqlite`。
- VPS 受控写入：160 buckets，276 moments，写入 12 条 `local_graph:`。
- `9be98de Preserve generated moment graph edges on refresh`：修复 `upsert_bucket()` / runtime refresh 会删掉 worker 边的问题。
- `server._refresh_moment_graph()` live 验证后仍有 `runtime_local_graph_edges: 12`，DB 里也仍是 12。
- `f151690 Add moment graph dry-run diagnostics`：worker 支持 `--diagnostics-file` JSONL 观察日志；VPS dry-run 结果为 160 buckets、276 moments、候选 12、写入 0，DB 仍是 12 条 `local_graph:`。
- VPS 已接 `ombre-moment-graph-dryrun.timer`：启动后 10 分钟、之后每 6 小时跑一次 dry-run 诊断，写 `/state/moment_graph_worker_diagnostics.jsonl`，不带 `--write`。

当前策略：

- 暂时不让 worker 自动写入。
- 如果后续要自动 `--write`，先观察 dry-run JSONL 里 `edge_fingerprint_changed`、候选数量和 sample_edges 是否稳定。

### 2. Typed edge + path scoring

第一版已完成，当前不是空 TODO。新增提交：

```text
442fd72 Add typed relation scoring for moment diffusion
034bc2e Tighten typed relation marker detection
08c8c2b Prefer old-version relation for digested graph moments
```

已实现：

- `memory_diffusion.py` 支持 `same_topic / cause / followup / embodiment_chain / conflict / old_version` 等 relation weights。
- `old_version` 和 `conflict` 默认降权；query 明确问旧版/冲突时，旧路径和冲突路径会被放开排序。
- `path_has_old_version()` 单独区分旧路径，Gateway / breath 不再把旧路径当普通背景。
- chain walk 不继续穿过 `old_version / conflict / contradicts / blocks`。
- worker 建边会从 section、facet、metadata 和强 marker 推断更细 relation type；`digested/resolved/archive` 优先归为 `old_version`。

已验证：

- 本地 `python -m pytest tests -q`：425 passed, 7 skipped。
- VPS 已部署到 `08c8c2b`，memory/gateway health 均 ok。
- VPS dry-run：160 buckets、276 moments、候选 12、写入 0；relation_counts 为 `old_version: 6`、`same_topic: 2`、`context_of: 4`。
- dry-run 后 DB 仍是 12 条 `local_graph:`，没有写库。

### 3. source_ref / transcript 行号

第一版已完成，先做的是 bucket markdown 行号，不是外部 transcript 行号。

新增提交：

```text
f375d06 Add source refs for memory moments
4e896b3 Clamp source ref windows to bucket content
```

已实现：

- `BucketManager` 读取 bucket 时记录 `path` 和正文起始行 `content_start_line`。
- `parse_bucket_moments()` 给 body / structured section moment 写入：

```yaml
source_ref:
  path: "/data/dynamic/xxx.md"
  content_start_line: 39
  start_line: 39
  end_line: 47
  source: "bucket_content"
```

- `source_ref_window()` 只读取 `allowed_root` 内文件，并用 `content_start_line` 防止窗口回卷到 frontmatter。
- Gateway / breath 的 direct 原文窗口在找不到 inline original text 时，会降级用 `source_ref_window()` 读 bucket 原文附近窗口。

当前召回语义：

- 可靠 direct 命中按 bucket 渲染：短桶原文、长桶 moment + 原文窗口、高价值/细节 query 脱水胶囊。
- 有 `source_ref` 时读取附近约 500 字证据窗。
- 没有 `source_ref` 时降级使用 MD moment 文本。

已验证：

- 本地 `python -m pytest tests -q`：428 passed, 7 skipped。
- VPS 已部署到 `4e896b3`，memory / gateway health 均 ok。
- VPS 刷新 moment 索引：160 buckets，276 parsed moments；DB 当前 294 moments，其中 263 条带 `source_ref`。
- 刷新前后 DB `local_graph:` 边都是 12，未丢 worker 边。
- VPS sample `source_ref_window()` 能读出 bucket 正文窗口，且不带 frontmatter。
- VPS worker dry-run：候选 12，写入 0；relation_counts 仍是 `old_version: 6`、`same_topic: 2`、`context_of: 4`。

仍未做：

- 外部 transcript / raw chat source 的行号导入。
- Dashboard 上展示 moment 对应原文窗口已在 2026-06-04 补上；Bucket 详情页 `Moments` 面板现在会显示 `source_window`。

### 4. Dashboard 观察面板

第一片已完成。新增提交：

```text
164c8a9 Show memory layer gates in breath dashboard
```

已实现：

- Breath 模拟结果每行显示 `layer`。
- 显示 related injection gate：允许/阻止和 reason。
- 显示 recent context gate：允许/阻止和 reason。
- query 需要 topic evidence 时显示 evidence 是否存在。

已验证：

- 本地 dashboard `<script>` 语法编译通过。
- 本地 `test_breath_debug_includes_runtime_gate` 通过。
- VPS 已部署到 `164c8a9`，memory / gateway health 均 ok。
- VPS `/dashboard` HTML 已包含 `score-trace` / `breathGateTrace`。
- VPS 容器内认证后调用 `/api/breath-debug?q=说教` 成功返回 layer/runtime gate 字段。

第二片已完成。新增：

- Dashboard bucket 详情页新增 `Moments` 只读面板。
- 新增 `/api/moments?bucket_id=...&limit=...`，复用 `inspect_moments` 的 payload，显示 bucket layer、moment layer、direct/related/context gate 和源正文行号。
- 该面板只刷新 `${state_dir}/memory_moments.sqlite` 观察索引，不修改 bucket，不改变 recall / gateway 注入行为。

第三片已完成。新增：

- Breath 模拟页新增 `扩散路径` 面板。
- 新增 `/api/diffusion-debug?q=...`，复用 `inspect_diffusion` 的 payload，显示 bucket-level seeds / hits / paths / gate。
- 这仍是 bucket-level diffusion 诊断，不等于完整 moment graph 观察面板。
- Breath 模拟页新增 `Moment 命中` 面板。
- 新增 `/api/recall-debug?q=...`，复用 breath 的 moment candidate / relevance gate / rerank / admission 逻辑，显示 query 命中哪个 moment、是否 admitted、是否 returned、被 gate 或 suppress 的原因。
- Bucket 详情页的 `Moments` 面板继续展示同 bucket 的 `moment_edges`：`source -> target`、relation、confidence、reason。
- Breath 模拟页新增 `Gateway 最近注入` 面板，通过 Dashboard-auth 的 `/api/gateway-injections` 代理读取 Gateway `/api/debug/injections`；默认不带完整 context，避免面板太吵。

当前优先级低于真实召回体验调参。

## 给下个窗口的接手顺序

1. 先读本文件。
2. 看当前 `git status --short --branch`。
3. 跑：

```powershell
python -m pytest tests/test_gateway.py tests/test_breath_edges.py tests/test_memory_recall_golden.py tests/test_memory_diffusion.py -q --tb=short
python -m py_compile gateway.py server.py memory_relevance.py memory_diffusion.py
```

4. 共享 query planner 已有 `RecallPolicy.plan_query()`；若继续改 recall gate，优先扩这个入口，不要让 `server.py` / `gateway.py` 重新分叉。
5. 若要排查 Gateway 注入，先看 `GET /api/debug/injections`。
6. 若要继续做图结构，再做本地 `moment graph worker`，不要把 LLM 建边塞进 `breath()`。
7. 若要调 profile_fact，先用 `introspection(created_date="YYYY-MM-DD")` 找证据桶，再手动调用 `profile_fact(...)`。

## 2026-06-03 追加：direct 返回形状先改，不先改召回轴

小雨重新评估后，决定不要先退回“整桶召回”的旧模式，也不要马上做完整 `retrieval_mode=bucket`。当前下一步先只改 Direct / Recalled 的展示形状，保留 p0 现有图召回、moment seed 和扩散约束。

### 目标行为

- `Direct / Recalled` 是证据层：可靠直接命中要比现在的 moment 摘要保留更多原文细节。
- 短桶：direct 返回整桶原文。
- 长桶：direct 返回命中 moment + 原文窗口。
- 高价值桶或用户明确问细节、原文、当时怎么说：长桶也返回脱水整桶胶囊。
- `Diffused / Related` 是联想层：永远只给摘要和路径，不搬远处整桶原文。
- `Dream` 是浮现层：永远返回梦境原文，不在 memory 层主动截断；只受模型上下文窗口影响。

### direct_render_mode 配置轴

已完成。新增提交：

```text
e26afa8 Add auto direct bucket rendering
f375d06 Add source refs for memory moments
4e896b3 Clamp source ref windows to bucket content
```

当前支持三档：

```yaml
gateway:
  direct_render_mode: auto   # auto | compact | full
```

含义：

- `auto`：默认。短桶原文；长桶 moment + 原文窗口；高价值桶或细节 query 用脱水整桶胶囊。
- `compact`：长桶只给命中 moment + 原文窗口，尽量少塞。
- `full`：可靠 direct 尽量整桶，太长再脱水。

`breath()` 也提供同名参数，默认 `auto`，让 MCP 和 Gateway 的直命中展示规则一致。

### retrieval_mode 配置轴

已完成，默认仍是 `graph`。

配置：

```yaml
gateway:
  retrieval_mode: graph   # graph | bucket
```

- `graph`：默认。p0 当前路线，moment + edges + diffusion。
- `bucket`：接近 main 的旧桶召回味道；不刷新 moment graph，不输出 `Diffused Memory`，只把可靠直命中的 bucket 按 `direct_render_mode` 渲染。
- `breath(query=..., retrieval_mode="bucket")` 支持同名参数，方便本地和 MCP 侧对照测试。

已验证：

- `breath(retrieval_mode="bucket")` 返回 `Recalled Memory`，不输出 `联想浮现`，不写 `memory_moments.sqlite`。
- Gateway `gateway.retrieval_mode: bucket` 在 related budget 和 memory edge 存在时仍不输出 `Diffused Memory`，不刷新 moment graph。

### 实现边界

- `graph` 模式下，moment 仍用于召回判定、可靠性 gate、扩散 seed。
- `bucket` 模式下，bucket 级候选仍要过可靠性门；只临时解析当前 bucket 用于 direct 渲染，不写索引。
- direct 最终展示按 bucket 渲染；同一 bucket 多个 moment 命中时只展示一次。
- 脱水胶囊应单独 cache，避免和扩散摘要的短 summary 混用。
- 扩散输出继续用 compact summary，不能因为 direct 改成原文而变吵。

当前剩余更像观察和体验调参，不是主链路缺口。

- Dashboard 只读观察面板已能看 query 命中 moment、direct / secondary 角色、direct 渲染形状、diffusion path、被 gate 掉的原因。
- 外部 transcript / raw chat source 若要接入，再把 `source_ref.source` 从 `bucket_content` 扩到 transcript 文件。

## 2026-06-04 追加：Dashboard 可调召回轴与 Gateway 热更新

新增提交：

```text
1763f9c Show memory diffusion config in dashboard
df0a41d Expose recall modes in dashboard config
5f045bb Hot reload memory diffusion config
```

当前 Dashboard 的“配置 -> 记忆浮现”可以直接调：

- Gateway 冷却：`cooldown_hours`、`skip_recent_rounds`
- Recent Context：`recent_context_cooldown_hours`、`recent_context_reentry_idle_hours`、`recent_context_budget`
- Recalled/Diffused 预算：`recalled_memory_budget`、`related_memory_budget`
- Query 旧记忆随机回响：`recall.query_resurface_enabled`
- 直命中展示：`direct_render_mode=auto|compact|full`
- 召回对照档：`retrieval_mode=graph|bucket`
- 图扩散：`memory_diffusion.enabled/top_k/min_activation`
- 链式扩散：`chain_walk_enabled/chain_max_hops/chain_min_confidence/chain_max_frontier`

`/api/config` 保存时会：

1. 更新 `ombre-brain` 当前进程里的 `config`。
2. `persist=true` 时写入 `config.yaml`，如果只读则同步到 `config.runtime.yaml`。
3. 如果配置了 `OMBRE_GATEWAY_ADMIN_URL`，把 `gateway` 和 `memory_diffusion` payload 一起 POST 到 `ombre-gateway /api/config`，让 Gateway 不重启就重算 `direct_render_mode/retrieval_mode/diffusion_options`，也立即更新 Recent Context 冷却、再进入阈值和预算。

已验证：

- 本地全量测试：`447 passed, 7 skipped`。
- VPS `/opt/Ombre-Brain` 已部署到 `5f045bb`。
- `http://8.136.154.242:18001/health` 和 `http://8.136.154.242:18002/health` 正常。
- 从 live `ombre-brain` 容器调用 `server.api_config_update()` 发送同值参数，返回 `gateway_hot_reloaded`，并包含 `memory_diffusion.top_k / chain_walk_enabled / chain_max_hops`。

## 2026-06-04 追加：query breath 默认不随机漂旧记忆

`breath(query=...)` 里的 `--- 久未碰过 ---` 随机旧记忆默认关闭。这样普通查询只返回可靠直命中、联想浮现和梦境，不会因为 direct 少于 3 条就突然混入无关旧桶。

如果确实想让有 query 的 breath 也随机回响旧记忆，可以显式设置：

```yaml
recall:
  query_resurface_enabled: true
```

更推荐的旧记忆抽卡入口仍是 `resurface(max_results=..., include_archive=...)`。

## 2026-06-04 追加：Dashboard 可调 Recent Context 阀门

`Recent Context` 的三个 Gateway 参数已接入 Dashboard 配置页和 `/api/config` 热更新：

- `recent_context_cooldown_hours`：最近上下文自动注入后的冷却，默认 `6`。
- `recent_context_reentry_idle_hours`：闲置多久算长时间再进入，默认 `24`；设 `0` 关闭再进入触发。
- `recent_context_budget`：最近上下文预算，默认 `300`；设 `0` 关闭这块自动注入。

这组参数只控制 `Recent Context`，不会影响可靠直命中 `Recalled Memory` 和扩散 `Diffused Memory`。

## 2026-06-04 追加：Dashboard 可调 query 旧记忆随机回响

`recall.query_resurface_enabled` 已接入 Dashboard 配置页和 `/api/config` 持久化。默认仍是 `false`：

- `false`：有 query 的 `breath()` 只返回可靠直命中、联想浮现和梦境，不随机追加 `--- 久未碰过 ---`。
- `true`：恢复旧行为，直命中稀疏且没有 related 时，query breath 可以按概率追加久未碰过的旧记忆。

这不是 Gateway 注入参数，保存后 `breath()` 当前进程立即生效；旧记忆抽卡仍更推荐显式调用 `resurface()`。

## 2026-06-04 追加：Dashboard 可调 Recalled/Diffused 预算

Gateway 的两块动态记忆预算已接入 Dashboard 配置页、Gateway `/api/config` 热更新和 memory service `/api/config` 转发：

- `recalled_memory_budget`：`Recalled Memory` 直命中预算，默认 `400`。
- `related_memory_budget`：`Diffused Memory` 扩散背景预算，默认 `220`；设 `0` 可以关闭 Gateway 扩散注入。

这只改变 Gateway 注入预算，不改 direct admission、diffusion gate 或 `breath()` 的 MCP 参数。

## 2026-06-04 追加：Bucket Moments 原文窗口

Dashboard Bucket 详情页的 `Moments` 面板现在会在每个 moment 下方显示只读 `source_window`：

- `/api/moments?bucket_id=...` 在 bucket 模式下给 moment payload 附带 `source_window`。
- `source_window` 通过 `source_ref_window()` 从 `${buckets_dir}` 内读取，只显示 bucket 正文附近窗口，不回卷到 frontmatter，也不会读出 bucket 根目录外的文件。
- 这只补观察面板，不改变 `breath()`、Gateway 注入、moment graph 或扩散规则。

## 2026-06-04 追加：单独情绪词不触发自动召回

`RecallPolicy.is_auto_query_too_vague()` 现在会把只有情绪、没有对象/事件/关系锚点的 query 当成 auto vague，例如：

- `开心^^`
- `我有点难过。`

影响范围：

- Gateway 自动注入不会因为 `开心/难过/累/焦虑` 这类单独情绪词命中旧 bucket。
- `breath(query=..., surface="auto")` 同样返回 `没有找到可靠命中。`
- 这不影响带具体对象的短 query，例如 `少女暴君`、`今天猫咪药量`。
- 也不是禁用情绪类记忆；如果 query 里有具体对象/事件，仍可正常召回。
