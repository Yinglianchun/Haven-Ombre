# 2026-06-02 可靠链式联想浮现

## 背景

旧版扩散主要靠 `max_hops` 固定半径：默认从 seed 走 1 跳，再走 2 跳，然后按 `top_k` 返回。这很安静，但会漏掉“蓝色偏好 -> 当时证据 -> 自我反思 -> 后续写入”这类可靠前后文链。

本次改动保留原默认行为，同时新增可选的 reliable chain walk。它不把更多候选塞成 seed，只改变 seed 之后怎么沿边走。

## 当前规则

入口仍是 `memory_diffusion.diffuse_memory()`。

`chain_walk_enabled=false` 时：

- 行为保持旧版：`max_hops` 控制扩散半径。
- `breath()` 的 related memory 仍强制一跳。

`chain_walk_enabled=true` 时：

- `chain_max_hops` 只是防爆上限，不再是主要语义边界。
- 每一步先算 activation；低于 `min_activation` 的节点不进入最终结果，也不继续扩散。
- 只有满足这些条件的路径会继续向后走：
  - path strength >= `chain_min_strength`
  - 当前边 confidence >= `chain_min_confidence`
  - 当前关系是可靠叙事/证据边，或 relation display priority >= `chain_min_relation_priority`
  - `contradicts` / `blocks` 只可作为谨慎背景命中，不继续扩散
- `relates_to` 这类泛关系可以作为近邻命中，但默认不会继续带出更远节点。
- `top_k` 仍控制最终返回条数。
- chain 模式最终排序优先看关系可靠度；可靠度相同的时候，离 seed 更近的节点排在更远的节点前面。

默认可继续扩展的关系：

- `same_event`
- `context_of`
- `precedes`
- `previous_context`
- `next_context`
- `updates`
- `evidenced_by`
- `reflects_on`

## Seed 边界

这次没有改 admission：

- `breath()` 仍只从已经展示的 direct bucket 做 related 扩散。
- hidden / suppressed direct candidates 不会成为 seed。
- Gateway 仍只从 admitted recalled moments 做 `Diffused Memory`。
- comments、affect anchors、favorite_reason 仍是温度上下文，不是 seed。

## 推荐配置

```yaml
memory_diffusion:
  enabled: true
  max_hops: 2
  top_k: 4
  min_activation: 0.18
  chain_walk_enabled: true
  chain_max_hops: 6
  chain_min_strength: 0.2
  chain_min_confidence: 0.72
  chain_min_relation_priority: 60
  chain_max_frontier: 24
```

Dashboard 的“配置 -> 记忆浮现”可以直接调整图扩散和链式扩散参数。保存后 `ombre-brain` 立即使用新值；如果配置了 `OMBRE_GATEWAY_ADMIN_URL`，Gateway 也会通过 `/api/config` 热更新 `diffusion_options`，不需要重启容器。

## 适合的边

手写或 backfill 关系边时，优先用这些关系：

- 同一事件的不同切片：`same_event`
- 前情/语境：`context_of`
- 时间先后：`precedes` / `previous_context` / `next_context`
- 证据来源：`evidenced_by`
- 事后反思：`reflects_on`
- 后续更新：`updates`

泛泛相似不要写成 `context_of`。如果只是“也相关”，用 `relates_to`，它默认不会拖出长链。
