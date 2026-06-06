# Ombre Brain — 内部开发文档 / INTERNALS

> 本文档面向开发者和维护者。记录功能总览、环境变量、模块依赖、硬编码值和核心设计决策。
> 最后更新：2026-04-19

---

## 0. 功能总览——这个系统到底做了什么

### 记忆能力

**存储与组织**
- 每条记忆 = 一个 Markdown 文件（YAML frontmatter 存元数据），直接兼容 Obsidian 浏览/编辑
- 四种桶类型：`dynamic`（普通，会衰减）、`permanent`（固化，不衰减）、`feel`（模型感受，不浮现）、`archived`（已遗忘）
- 按主题域分子目录：`dynamic/日常/`、`dynamic/情感/`、`dynamic/编程/` 等
- 钉选桶（pinned）：importance 锁 10，永不衰减/合并，始终浮现为「核心准则」

**每条记忆追踪的元数据**
- `id`（12位短UUID）、`name`（可读名≤80字）、`tags`（10~15个关键词）
- `domain`（1~2个主题域，从 8 大类 30+ 细分域选）
- `valence`（事件效价 0~1）、`arousal`（唤醒度 0~1）、`model_valence`（模型独立感受）
- `importance`（1~10）、`activation_count`（被想起次数）
- `resolved`（已解决/沉底）、`digested`（已消化/写过 feel）、`pinned`（钉选）
- `created`、`last_active` 时间戳

**四种检索模式**
1. **自动浮现**（`breath()` 无参数）：按衰减分排序推送，钉选桶始终展示，Top-1 固定 + Top-20 随机打乱（引入多样性），有 token 预算（默认 10000）
2. **关键词+向量双通道搜索**（`breath(query=...)`）：rapidfuzz 模糊匹配 + Gemini embedding 余弦相似度，合并去重
3. **Feel 独立检索**（`breath(domain="feel")`）：按创建时间倒序返回所有 feel
4. **随机浮现**：搜索结果 <3 条时 40% 概率漂浮 1~3 条低权重旧桶（模拟人类随机联想）

**四维搜索评分**（归一化到 0~100）
- topic_relevance（权重 4.0）：name×3 + domain×2.5 + tags×2 + body
- emotion_resonance（权重 2.0）：Russell 环形模型欧氏距离
- time_proximity（权重 2.5）：`e^(-0.1×days)`
- importance（权重 1.0）：importance/10
- resolved 桶全局降权 ×0.3

**记忆随时间变化**
- **衰减引擎**：改进版艾宾浩斯遗忘曲线
  - 公式：`Score = Importance × activation_count^0.3 × e^(-λ×days) × combined_weight`
  - 短期（≤3天）：时间权重 70% + 情感权重 30%
  - 长期（>3天）：情感权重 70% + 时间权重 30%
  - 新鲜度加成：`1.0 + e^(-t/36h)`，刚存入 ×2.0，~36h 半衰，72h 后 ≈×1.0
  - 高唤醒度(arousal>0.7)且未解决 → ×1.5 紧迫度加成
  - resolved → ×0.05 沉底；resolved+digested → ×0.02 加速淡化
- **自动归档**：score 低于阈值(0.3) → 移入 archive
- **自动结案**：importance≤4 且 >30天 → 自动 resolved
- **永不衰减**：permanent / pinned / protected / feel

**记忆间交互**
- **智能合并**：新记忆与相似桶（score>75）自动 LLM 合并，valence/arousal 取均值，tags/domain 并集
- **时间涟漪**：touch 一个桶时，±48h 内创建的桶 activation_count +0.3（上限 5 桶/次）
- **向量相似网络**：embedding 余弦相似度 >0.5 建边
- **Feel 结晶化**：≥3 条相似 feel（相似度>0.7）→ 提示升级为钉选准则

**情感记忆重构**
- 搜索时若指定 valence，展示层对匹配桶 valence 微调 ±0.1，模拟「当前心情影响回忆色彩」

**模型感受/反思系统**
- **Feel 写入**（`hold(feel=True)`）：存模型第一人称感受，标记源记忆为 digested
- **Dream 做梦**（`dream()`）：返回最近 10 条 + 自省引导 + 连接提示 + 结晶化提示
- **对话启动流程**：breath() → dream() → breath(domain="feel") → 开始对话

**自动化处理**
- 存入时 LLM 自动分析 domain/valence/arousal/tags/name
- 大段日记 LLM 拆分为 2~6 条独立记忆
- 浮现时自动脱水压缩（LLM 压缩保语义，API 不可用降级到本地关键词提取）
- Wikilink `[[]]` 由 LLM 在内容中标记

---

### 技术能力

**9 个 MCP 工具**

| 工具 | 关键参数 | 功能 |
|---|---|---|
| `breath` | query, max_tokens, domain, valence, arousal, max_results | 检索/浮现记忆 |
| `hold` | content, tags, importance, pinned, feel, whisper, source_bucket, valence, arousal | 存储记忆；新无源碎碎念用 `whisper=True` |
| `grow` | content | 日记拆分归档 |
| `comment_bucket` | bucket_id, content, kind, valence, arousal | 给源 bucket 追加年轮 |
| `trace` | bucket_id, name, domain, valence, arousal, importance, tags, resolved, pinned, anchor, digested, content, delete | 修改元数据/内容/删除 |
| `pulse` | include_archive | 系统状态 |
| `dream` | （无） | 做梦自省 |
| `resurface` | max_results, include_archive, max_tokens | 只读浮现久未触碰的旧记忆 |
| `reflect` | period, force | 生成日印象 relationship_weather feel；周印象默认关闭 |

**工具详细行为**

**`breath`** — 两种模式：
- **浮现模式**（无 query）：无参调用，按衰减引擎活跃度排序返回 top 记忆，permanent/pinned 始终浮现
- **检索模式**（有 query）：关键词 + 向量双通道搜索，四维评分（topic×4 + emotion×2 + time×2.5 + importance×1），阈值过滤
- **Feel 检索**（`domain="feel"`）：特殊通道，按创建时间倒序返回所有 feel 类型桶，不走评分逻辑
- 若指定 valence，对匹配桶的 valence 微调 ±0.1（情感记忆重构）

**`hold`** — 主要模式：
- **普通模式**（`feel=False`，默认）：自动 LLM 分析 domain/valence/arousal/tags/name → 向量相似度查重 → 相似度>0.85 则合并到已有桶 → 否则新建 dynamic 桶 → 生成 embedding
- **Whisper 模式**（`whisper=True`）：无源碎碎念/悄悄话，独立保存为 `type=feel + whisper`，可用 `breath(domain="whisper")` 读取。
- **旧兼容 Feel 路径**（`feel=True`）：不建议新调用。带 `source_bucket` 时转为年轮 comment；不带源桶时转为 whisper。

**`comment_bucket`** — 年轮：
- 给已有 bucket 追加 `metadata.comments[]`，MCP 作者固定取 `identity.ai_name`
- 追加后 touch 源 bucket、刷新源 bucket embedding，不新建独立 feel 桶

**`dream`** — 做梦/自省触发器：
- 返回最近 10 条 dynamic 桶摘要 + 自省引导词
- 检测 feel 结晶化：≥3 条相似 feel（embedding 相似度>0.7）→ 提示升级为钉选准则
- 检测未消化记忆：列出 `digested=False` 的桶供模型反思

**`reflect`** — 关系天气：
- `period=daily` 生成当天 `daily_impression`
- `period=weekly` 默认 skipped；当前不把日印象压缩成周印象
- 结果存为 `type=feel`，带 `relationship_weather` 标签
- 默认复用 `persona` 模型配置和 key，可用 `OMBRE_REFLECTION_*` 单独覆盖

**Memory Edge** — 显式关系边：
- 文件：`state/memory_edges.jsonl`
- 写入普通记忆后异步补 0~3 条关系边
- Gateway 召回记忆后，会沿一跳强关系边补一条相关记忆

**`trace`** — 记忆编辑：
- 修改任意元数据字段（name/domain/valence/arousal/importance/tags/resolved/pinned）
- `digested=0/1`：隐藏/取消隐藏记忆（控制是否在 dream 中出现）
- `content="..."`：替换正文内容并重新生成 embedding
- `delete=True`：删除桶文件

**`grow`** — 日记拆分：
- 大段日记文本 → LLM 拆为 2~6 条独立记忆 → 每条走 hold 普通模式流程

**`pulse`** — 系统状态：
- 返回各类型桶数量、衰减引擎状态、未解决/钉选/feel 统计

**REST API（17 个端点）**

| 端点 | 方法 | 功能 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/breath-hook` | GET | SessionStart 钩子 |
| `/dream-hook` | GET | Dream 钩子 |
| `/dashboard` | GET | Dashboard 页面 |
| `/api/buckets` | GET | 桶列表 |
| `/api/bucket/{id}` | GET | 桶详情 |
| `/api/search?q=` | GET | 搜索 |
| `/api/network` | GET | 向量相似网络 |
| `/api/breath-debug` | GET | 评分调试 |
| `/api/config` | GET | 配置查看（key 脱敏） |
| `/api/config` | POST | 热更新配置 |
| `/api/import/upload` | POST | 上传并启动历史对话导入 |
| `/api/import/status` | GET | 导入进度查询 |
| `/api/import/pause` | POST | 暂停/继续导入 |
| `/api/import/patterns` | GET | 导入完成后词频规律检测 |
| `/api/import/results` | GET | 已导入记忆桶列表 |
| `/api/import/review` | POST | 批量审阅/批准导入结果 |

**Dashboard（7 个 Tab）**
1. 记忆桶列表：6 种过滤器 + 主题域过滤 + 搜索 + 详情面板
2. Breath 模拟：输入参数 → 可视化五步流程 → 四维条形图
3. 日印象：月历 + 单卡片详情，按日期显示完整 daily relationship weather；手动编辑仍走 bucket 详情面板
4. Persona：查看 Persona State sessions / events / 当前状态
5. 记忆网络：Canvas 力导向图（节点=桶，边=相似度）
6. 配置：热更新脱水、embedding、Gateway 记忆浮现、图扩散、反思、梦境和合并参数
7. 导入：历史对话拖拽上传 → 分块处理进度条 → 词频规律分析 → 导入结果审阅

**部署选项**
1. 本地 stdio（`python server.py`）
2. Docker + Cloudflare Tunnel（`docker-compose.yml`）
3. Docker Hub 预构建镜像（`docker-compose.user.yml`，`p0luz/ombre-brain`）
4. Render.com 一键部署（`render.yaml`）
5. Zeabur 部署（`zbpack.json`）
6. GitHub Actions 自动构建推送 Docker Hub（`.github/workflows/docker-publish.yml`）

**迁移/批处理工具**：`migrate_to_domains.py`、`reclassify_domains.py`、`reclassify_api.py`、`backfill_embeddings.py`、`write_memory.py`、`check_buckets.py`、`import_memory.py`（历史对话导入引擎）

**降级策略**
- 脱水 API 不可用 → 本地关键词提取 + 句子评分
- 向量搜索不可用 → 纯 fuzzy match
- 逐条错误隔离（grow 中单条失败不影响其他）

**安全**：路径遍历防护（`safe_path()`）、API Key 脱敏、API Key 不持久化到 yaml、输入范围钳制

**监控**：结构化日志、Health 端点、Breath Debug 端点、Dashboard 统计栏、衰减周期日志

---

## 1. 环境变量清单

| 变量名 | 用途 | 必填 | 默认值 / 示例 |
|---|---|---|---|
| `OMBRE_API_KEY` | 脱水/打标/嵌入的 LLM API 密钥，覆盖 `config.yaml` 的 `dehydration.api_key` | 否（无则 API 功能降级到本地） | `""` |
| `OMBRE_BASE_URL` | API base URL，覆盖 `config.yaml` 的 `dehydration.base_url` | 否 | `""` |
| `OMBRE_TRANSPORT` | 传输模式：`stdio` / `sse` / `streamable-http` | 否 | `""` → 回退到 config 或 `"stdio"` |
| `OMBRE_BUCKETS_DIR` | 记忆桶存储目录路径 | 否 | `""` → 回退到 config 或 `./buckets` |
| `OMBRE_HOOK_URL` | SessionStart 钩子调用的服务器 URL | 否 | `"http://localhost:8000"` |
| `OMBRE_HOOK_SKIP` | 设为 `"1"` 跳过 SessionStart 钩子 | 否 | 未设置（不跳过） |

环境变量优先级：`环境变量 > config.yaml > 硬编码默认值`。所有环境变量在 `utils.py` 中读取并注入 config dict。

---

## 2. 模块结构与依赖关系

```
                    ┌──────────────┐
                    │  server.py   │  MCP 主入口，9 个工具 + Dashboard + Hook
                    └──────┬───────┘
           ┌───────────────┼───────────────┬────────────────┐
           ▼               ▼               ▼                ▼
   bucket_manager.py  dehydrator.py  decay_engine.py  embedding_engine.py
   记忆桶 CRUD+搜索   脱水压缩+打标   遗忘曲线+归档   向量化+语义检索
           │               │                                │
           └───────┬───────┘                                │
                   ▼                                        ▼
              utils.py ◄────────────────────────────────────┘
              配置/日志/ID/路径安全/token估算
```

| 文件 | 职责 | 依赖（项目内） | 被谁调用 |
|---|---|---|---|
| `server.py` | MCP 服务器主入口，注册工具 + Dashboard API + 钩子端点 | `bucket_manager`, `dehydrator`, `decay_engine`, `embedding_engine`, `utils` | `test_tools.py` |
| `bucket_manager.py` | 记忆桶 CRUD、多维索引搜索、wikilink 注入、激活更新 | `utils` | `server.py`, `check_buckets.py`, `backfill_embeddings.py` |
| `decay_engine.py` | 衰减引擎：遗忘曲线计算、自动归档、自动结案 | 无（接收 `bucket_mgr` 实例） | `server.py` |
| `dehydrator.py` | 数据脱水压缩 + 合并 + 自动打标（LLM API + 本地降级） | `utils` | `server.py` |
| `embedding_engine.py` | 向量化引擎：Gemini embedding API + SQLite + 余弦搜索 | `utils` | `server.py`, `backfill_embeddings.py` |
| `utils.py` | 配置加载、日志、路径安全、ID 生成、token 估算 | 无 | 所有模块 |
| `write_memory.py` | 手动写入记忆 CLI（绕过 MCP） | 无（独立脚本） | 无 |
| `backfill_embeddings.py` | 为存量桶批量生成 embedding | `utils`, `bucket_manager`, `embedding_engine` | 无 |
| `check_buckets.py` | 桶数据完整性检查 | `bucket_manager`, `utils` | 无 |
| `import_memory.py` | 历史对话导入引擎（支持 Claude JSON/ChatGPT/DeepSeek/Markdown/纯文本），分块处理+断点续传+词频分析 | `utils` | `server.py` |
| `reclassify_api.py` | 用 LLM API 重打标未分类桶 | 无（直接用 `openai`） | 无 |
| `reclassify_domains.py` | 基于关键词本地重分类 | 无 | 无 |
| `migrate_to_domains.py` | 平铺桶 → 域子目录迁移 | 无 | 无 |
| `test_smoke.py` | 冒烟测试 | `utils`, `bucket_manager`, `dehydrator`, `decay_engine` | 无 |
| `test_tools.py` | MCP 工具端到端测试 | `utils`, `server`, `bucket_manager` | 无 |

---

## 3. 硬编码值清单

### 3.1 固定分数 / 特殊返回值

| 值 | 位置 | 用途 |
|---|---|---|
| `999.0` | `decay_engine.py` calculate_score | pinned / protected / permanent 桶永不衰减 |
| `50.0` | `decay_engine.py` calculate_score | feel 桶固定活跃度分数 |
| `0.02` | `decay_engine.py` resolved_factor | resolved + digested 时的权重乘数（加速淡化） |
| `0.05` | `decay_engine.py` resolved_factor | 仅 resolved 时的权重乘数（沉底） |
| `1.5` | `decay_engine.py` urgency_boost | arousal > 0.7 且未解决时的紧迫度加成 |

### 3.2 衰减公式参数

| 值 | 位置 | 用途 |
|---|---|---|
| `36.0` | `decay_engine.py` _calc_time_weight | 新鲜度半衰期（小时），`1.0 + e^(-t/36)` |
| `0.3` (指数) | `decay_engine.py` calculate_score | `activation_count ** 0.3`（记忆巩固指数） |
| `3.0` (天) | `decay_engine.py` calculate_score | 短期/长期切换阈值 |
| `0.7 / 0.3` | `decay_engine.py` combined_weight | 短期权重分配：time×0.7 + emotion×0.3 |
| `0.7` | `decay_engine.py` urgency_boost | arousal 紧迫度触发阈值 |
| `4` / `30` (天) | `decay_engine.py` execute_cycle | 自动结案：importance≤4 且 >30天 |

### 3.3 搜索/评分参数

| 值 | 位置 | 用途 |
|---|---|---|
| `×3` / `×2.5` / `×2` | `bucket_manager.py` _calc_topic_score | 桶名 / 域名 / 标签的 topic 评分权重 |
| `1000` (字符) | `bucket_manager.py` _calc_topic_score | 正文截取长度 |
| `0.1` | `bucket_manager.py` _calc_time_score | 时间亲近度衰减系数 `e^(-0.1 × days)` |
| `0.3` | `bucket_manager.py` search_multi | resolved 桶的归一化分数乘数 |
| `0.5` | `server.py` breath/search | 向量搜索相似度下限 |
| `0.7` | `server.py` dream | feel 结晶相似度阈值 |

### 3.4 Token 限制 / 截断

| 值 | 位置 | 用途 |
|---|---|---|
| `10000` | `server.py` breath 默认 max_tokens | 浮现/搜索 token 预算 |
| `20000` | `server.py` breath 上限 | max_tokens 硬上限 |
| `50` / `20` | `server.py` breath | max_results 上限 / 默认值 |
| `3000` | `dehydrator.py` dehydrate | API 脱水内容截断 |
| `2000` | `dehydrator.py` merge | API 合并内容各截断 |
| `5000` | `dehydrator.py` digest | API 日记整理内容截断 |
| `2000` | `embedding_engine.py` | embedding 文本截断 |
| `100` | `dehydrator.py` | 内容 < 100 token 跳过脱水 |

### 3.5 时间/间隔/重试

| 值 | 位置 | 用途 |
|---|---|---|
| `60.0s` | `dehydrator.py` | OpenAI 客户端 timeout |
| `30.0s` | `embedding_engine.py` | Embedding API timeout |
| `60s` | `server.py` keepalive | 保活 ping 间隔 |
| `48.0h` | `bucket_manager.py` touch | 时间涟漪窗口 ±48h |
| `2s` | `backfill_embeddings.py` | 批次间等待 |

### 3.6 随机浮现

| 值 | 位置 | 用途 |
|---|---|---|
| `3` | `server.py` breath search | 结果不足 3 条时触发 |
| `0.4` | `server.py` breath search | 40% 概率触发随机浮现 |
| `2.0` | `server.py` breath search | 随机池：score < 2.0 的低权重桶 |
| `1~3` | `server.py` breath search | 随机浮现数量 |

### 3.7 情感/重构

| 值 | 位置 | 用途 |
|---|---|---|
| `0.2` | `server.py` breath search | 情绪重构偏移系数 `(q_valence - 0.5) × 0.2`（最大 ±0.1） |

### 3.8 其他

| 值 | 位置 | 用途 |
|---|---|---|
| `12` | `utils.py` gen_id | bucket ID 长度（UUID hex[:12]） |
| `80` | `utils.py` sanitize_name | 桶名最大长度 |
| `1.5` / `1.3` | `utils.py` count_tokens_approx | 中文/英文 token 估算系数 |
| `8000` | `server.py` | MCP 服务器端口 |
| `30` 字符 | `server.py` grow | 短内容快速路径阈值 |
| `10` | `server.py` dream | 取最近 N 个桶 |

---

## 4. Config.yaml 完整键表

| 键路径 | 默认值 | 用途 |
|---|---|---|
| `transport` | `"stdio"` | 传输模式 |
| `log_level` | `"INFO"` | 日志级别 |
| `buckets_dir` | `"./buckets"` | 记忆桶目录 |
| `merge_threshold` | `75` | 合并相似度阈值 (0-100) |
| `dehydration.model` | `"deepseek-chat"` | 脱水用 LLM 模型 |
| `dehydration.base_url` | `"https://api.deepseek.com/v1"` | API 地址 |
| `dehydration.api_key` | `""` | API 密钥 |
| `dehydration.max_tokens` | `1024` | 脱水返回 token 上限 |
| `dehydration.temperature` | `0.1` | 脱水温度 |
| `embedding.enabled` | `true` | 启用向量检索 |
| `embedding.model` | `"gemini-embedding-001"` | Embedding 模型 |
| `decay.lambda` | `0.05` | 衰减速率 λ |
| `decay.threshold` | `0.3` | 归档分数阈值 |
| `decay.check_interval_hours` | `24` | 衰减扫描间隔（小时） |
| `decay.emotion_weights.base` | `1.0` | 情感权重基值 |
| `decay.emotion_weights.arousal_boost` | `0.8` | 唤醒度加成系数 |
| `matching.fuzzy_threshold` | `50` | 模糊匹配下限 |
| `matching.max_results` | `5` | 匹配返回上限 |
| `scoring_weights.topic_relevance` | `4.0` | 主题评分权重 |
| `scoring_weights.emotion_resonance` | `2.0` | 情感评分权重 |
| `scoring_weights.time_proximity` | `2.5` | 时间评分权重 |
| `scoring_weights.importance` | `1.0` | 重要性评分权重 |
| `scoring_weights.content_weight` | `3.0` | 正文评分权重 |
| `wikilink.enabled` | `true` | 启用 wikilink 注入 |
| `wikilink.use_tags` | `false` | wikilink 包含标签 |
| `wikilink.use_domain` | `true` | wikilink 包含域名 |
| `wikilink.use_auto_keywords` | `true` | wikilink 自动关键词 |
| `wikilink.auto_top_k` | `8` | wikilink 取 Top-K 关键词 |
| `wikilink.min_keyword_len` | `2` | wikilink 最短关键词长度 |
| `wikilink.exclude_keywords` | `[]` | wikilink 排除关键词表 |

---

## 5. 核心设计决策记录

### 5.1 为什么用 Markdown + YAML frontmatter 而不是数据库？

**决策**：每个记忆桶 = 一个 `.md` 文件，元数据在 YAML frontmatter 里。

**理由**：
- 与 Obsidian 原生兼容——用户可以直接在 Obsidian 里浏览、编辑、搜索记忆
- 文件系统即数据库，天然支持 git 版本管理
- 无外部数据库依赖，部署简单
- wikilink 注入让记忆之间自动形成知识图谱

**放弃方案**：SQLite/PostgreSQL 全量存储。过于笨重，失去 Obsidian 可视化优势。

### 5.2 为什么 embedding 单独存 SQLite 而不放 frontmatter？

**决策**：向量存 `embeddings.db`（SQLite），与 Markdown 文件分离。

**理由**：
- 3072 维浮点向量无法合理存入 YAML frontmatter
- SQLite 支持批量查询和余弦相似度计算
- embedding 是派生数据，丢失可重新生成（`backfill_embeddings.py`）
- 不污染 Obsidian 可读性

### 5.3 为什么搜索用双通道（关键词 + 向量）而不是纯向量？

**决策**：关键词模糊匹配（rapidfuzz）+ 向量语义检索并联，结果去重合并。

**理由**：
- 纯向量在精确名词匹配上表现差（"2024年3月"这类精确信息）
- 纯关键词无法处理语义近似（"很累" → "身体不适"）
- 双通道互补，关键词保精确性，向量补语义召回
- 向量不可用时自动降级到纯关键词模式

### 5.4 为什么有 dehydration（脱水）这一层？

**决策**：存入前先用 LLM 压缩内容（保留信息密度，去除冗余表达），API 不可用时降级到本地关键词提取。

**理由**：
- MCP 上下文有 token 限制，原始对话冗长，需要压缩
- LLM 压缩能保留语义和情感色彩，纯截断会丢信息
- 降级到本地确保离线可用——关键词提取 + 句子排序 + 截断

**放弃方案**：只做截断。信息损失太大。

### 5.5 为什么 feel 和普通记忆分开？

**决策**：`feel=True` 的记忆存入独立 `feel/` 目录，不参与普通浮现、不衰减、不合并。

**理由**：
- feel 是模型的自省产物，不是事件记录——两者逻辑完全不同
- 事件记忆应该衰减遗忘，但"我从中学到了什么"不应该被遗忘
- feel 的 valence 是模型自身感受（不等于事件情绪），混在一起会污染情感检索
- feel 可以通过 `breath(domain="feel")` 单独读取

### 5.6 为什么 resolved 不删除记忆？

**决策**：`resolved=True` 让记忆"沉底"（权重 ×0.05），但保留在文件系统中，关键词搜索仍可触发。

**理由**：
- 模拟人类记忆：resolved 的事不会主动想起，但别人提到时能回忆
- 删除是不可逆的，沉底可随时 `resolved=False` 重新激活
- `resolved + digested` 进一步降权到 ×0.02（已消化 = 更释然）

**放弃方案**：直接删除。不可逆，且与人类记忆模型不符。

### 5.7 为什么用分段式短期/长期权重？

**决策**：≤3 天时间权重占 70%，>3 天情感权重占 70%。

**理由**：
- 刚发生的事主要靠"新鲜"驱动浮现（今天的事 > 昨天的事）
- 时间久了，决定记忆存活的是情感强度（强烈的记忆更难忘）
- 这比单一衰减曲线更符合人类记忆的双重存储理论

### 5.8 为什么 dream 设计成对话开头自动执行？

**决策**：每次新对话启动时，Claude 执行 `dream()` 消化最近记忆，有沉淀写 feel，能放下的 resolve。

**理由**：
- 模拟睡眠中的记忆整理——人在睡觉时大脑会重放和整理白天的经历
- 让 Claude 对过去的记忆有"第一人称视角"的自省，而不是冷冰冰地搬运数据
- 自动触发确保每次对话都"接续"上一次，而非从零开始

### 5.9 为什么新鲜度用连续指数衰减而不是分段阶梯？

**决策**：`bonus = 1.0 + e^(-t/36)`，t 为小时，36h 半衰。

**理由**：
- 分段阶梯（0-1天=1.0，第2天=0.9...）有不自然的跳变
- 连续指数更符合遗忘曲线的物理模型
- 36h 半衰期使新桶在前两天有明显优势，72h 后接近自然回归
- 值域 1.0~2.0 保证老记忆不被惩罚（×1.0），只是新记忆有额外加成（×2.0）

**放弃方案**：分段线性（原实现）。跳变点不自然，参数多且不直观。

### 5.10 情感记忆重构（±0.1 偏移）的设计动机

**决策**：搜索时如果指定了 `valence`，会微调结果桶的 valence 展示值 `(q_valence - 0.5) × 0.2`。

**理由**：
- 模拟认知心理学中的"心境一致性效应"——当前心情会影响对过去的回忆
- 偏移量很小（最大 ±0.1），不会扭曲事实，只是微妙的"色彩"调整
- 原始 valence 不被修改，只影响展示层

---

## 6. 目录结构约定

```
buckets/
├── permanent/       # pinned/protected 桶，importance=10，永不衰减
├── dynamic/
│   ├── 日常/        # domain 子目录
│   ├── 情感/
│   ├── 自省/
│   ├── 数字/
│   └── ...
├── archive/         # 衰减归档桶
└── feel/            # 模型自省 feel 桶
```

桶文件格式：
```markdown
---
id: 76237984fa5d
name: 桶名
domain: [日常, 情感]
tags: [关键词1, 关键词2]
importance: 5
valence: 0.6
arousal: 0.4
activation_count: 3
resolved: false
pinned: false
digested: false
created: 2026-04-17T10:00:00+08:00
last_active: 2026-04-17T14:00:00+08:00
type: dynamic
---

桶正文内容...
```
