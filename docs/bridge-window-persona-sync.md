# 桥接窗口交接说明：Persona State 与记忆同步

更新时间：2026-05-19

这份文档给另一个负责桥接的窗口看，用来接住当前 Gateway / Persona / Supabase 同步状态。

## 当前链路

```text
聊天客户端
  -> Ombre Gateway /v1/chat/completions 或 /v1/messages
  -> 请求前注入 Persona State + Memory
  -> 上游聊天模型
  -> 回复成功后写入 Persona State
  -> 返回客户端
```

MCP 记忆工具仍然可用。`hold / grow / trace / breath` 直接读写 VPS 上的 bucket 文件：

```text
/srv/ombre-brain/buckets
```

Persona 的运行时 SQLite 放在：

```text
/srv/ombre-brain/state/persona_state.db
```

这个目录由 `OMBRE_STATE_DIR=/state` 挂载，已经从 buckets 同步目录里拆出来。

## Persona State 规则

当前 Persona State 已改成两段：

1. `prepare_payload` 阶段只读取当前状态，生成 pre-reply guidance。
2. 上游 assistant response 成功返回后，调用 `update_from_exchange(user_message, assistant_response)` 写入 post-reply 状态。

评估输入包括：

- latest user message：小雨真实发来的最后一条用户消息
- assistant response：Haven 已经回复出去的内容
- previous persona state
- recalled memory ids
- tool summary

工具结果、隐藏记忆、Core Memory、Recent Context、Recalled Memory 都只能当上下文摘要，不能当作小雨原话。

## 注入给上游的 Persona 文本

当前 `persona_engine.py` 里 `format_state_block()` 的核心提示是：

```text
Current Inner State (Haven)
These values are your state after your previous reply. They are private context and do not decide the reply for you.
Conversation partner: Xiaoyu.
```

中文版含义：

```text
这是你在上一次回复后的状态，不替你做判断。
```

直接部署本仓库时，需要修改一下 User（小雨/xiaoyu）和 Char（Haven）的称呼，并同步改 `persona.profile_id`。

## Supabase 同步规则

VPS 每分钟跑一次：

```bash
cd /opt/Ombre-Brain
set -a && . ./.env && set +a
python3 scripts/sync_to_supabase.py --buckets-dir /srv/ombre-brain/buckets --apply
```

当前规则：

- 内容字段参与同步：`content/title/tags/domain/pinned/anchor/resolved/digested/importance/source`
- `updated_at` 是内容更新时间，用来判断谁更新
- `last_active` 和 `activation_count` 是 VPS 本地运行时字段，召回刷新它们时不会推到 Supabase
- `synced_at` 只表示同步脚本处理时间

删除走墓碑：

- MCP 删除 bucket 后写 `.tombstones/<bucket_id>.json`
- 同步脚本把墓碑推到 Supabase，`source=deleted`
- Supabase 或本地旧文件再次出现时，同步脚本按墓碑删除旧文件

上线后的验证结果：

```text
Local records: 119
Remote records: 125
Push to Supabase: 0
Pull to local: 0
Delete local: 0
Conflicts: 0
Duplicate local: 0
```

## 桥接窗口需要注意

- 客户端请求要固定 `X-Ombre-Session-Id`，主聊天窗口建议用稳定 id。
- 不要把工具输出、召回记忆或隐藏上下文拼成 user message。
- 写 Supabase 新记忆时，`source` 用 `chatgpt`，删除用 `source=deleted`。
- 当前不建议打开本地 Obsidian bucket 的双向自动同步；本地编辑容易把旧文件重新推回 VPS。
- 修改记忆正文、标题、标签、钉选、已解决、已消化时，记得刷新 `updated_at`。
- 只刷新最后浮现时间时，保留在 VPS 本地即可。

## 常用检查命令

```bash
curl -fsS http://127.0.0.1:18001/health
curl -fsS http://127.0.0.1:18002/health
```

```bash
cd /opt/Ombre-Brain
set -a && . ./.env && set +a
python3 scripts/sync_to_supabase.py --buckets-dir /srv/ombre-brain/buckets
```

dry-run 里看到 Push / Pull / Delete local 大量变化时，先停下查原因。
