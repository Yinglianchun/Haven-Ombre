# 2026-06-05 Memory System Next Plan

## Goal

This is not a rewrite. The goal is to make the existing memory stack easier to reason about:

- Long natural messages should recall better when they mix emotion, background, current task, metaphors, names, and casual side mentions.
- Long-term user context should have a clear place that is separate from Core Memory and dynamic recall.
- LLM-maintained memory should stay evidence-bound: it may propose updates, but it must not silently rewrite roots.

## Current Facts

- Gateway currently uses the current user message as one recall query in normal bucket mode or graph mode.
- The existing recall path already has deterministic term extraction, context-term filtering, facet expansion, topic evidence, vague gates, budget, cooldown, and `RecallPolicy`.
- Gateway now has an optional LLM query planner that splits long mixed messages into short search anchors.
- `core_memory_interval_rounds` defaults to `0`, so Core Memory injection is off unless configured.
- When enabled, Gateway's Core Memory block only takes `pinned` or `protected` buckets.
- `memory_layers.py` classifies ordinary `permanent` buckets as core layer, but Gateway does not inject ordinary permanent buckets into the Core Memory block.

This means the next change should clarify boundaries before adding more automatic behavior.

## 2026-06-06 Status Update

The current slice is complete on `feature/memory-diffusion-p0`.

- Private Alias is live as a deployment-private overlay, not a template feature. The VPS uses `/state/private_identity_semantics.yaml`, derived from the previous `reflection.identity_role_edges`. `config.example.yaml` keeps `identity_semantics` empty. The rebuilt live index has `8` canonical nodes, `12` aliases, and `21` evidence links.
- Word Map Lite and private identity diagnostics are implemented as dashboard/manual rebuild tooling. They do not inject anything into Gateway by themselves.
- `scripts/migrate_affect_anchor_sections.py` migrates old bucket structure without rewriting memory meaning:
  - facts move to `### moment`
  - Haven interpretations such as `Haven 由此确认...` move to `### assistant_reflection`
  - `### affect_anchor` keeps only chords, `含义：...`, and temperature/poetic markers
  - duplicate short facts are skipped when already covered by longer body text
- The migration also handles the unheaded-body case where a reflection paragraph was written in the main body without a `### assistant_reflection` heading.
- Live VPS migration was applied after dry-run review:
  - first `2c4b82ee93ba` and `reflection_daily_2026-06-05`
  - then `65023203392f`
  - then the remaining 53 default-scope buckets
- Final verification:
  - default dry-run: `0` remaining
  - `--include-archive` dry-run: `0` remaining
  - every applied bucket reported `written=true`, `embedding_refreshed=true`, and `moment_index_refreshed=true`
  - `breath(query="激动哭")` hits `2c4b82ee93ba`
  - `breath(query="记忆改版 模型更新")` hits `65023203392f` and returns the new `### assistant_reflection`

## Memory Boundaries

### `pinned` / `protected`

Root settings and non-negotiable continuity. These are not maintained by the LLM automatically.

They can appear in Gateway's Core Memory block when Core Memory injection is enabled.

### Ordinary `permanent`

Long-lived stored memory, but not automatically part of Gateway Core Memory.

Do not treat every `permanent` bucket as root context when building Portrait Memory.

### `profile_fact`

Evidence-bound user portrait facts. These should remain factual, inspectable, and editable.

Each fact must keep an evidence bucket or evidence moment. No evidence means no write.

### `anchor`

Long-term important experiences, relationships, or recurring life facts.

Anchors can help Portrait Memory, but they should still pass the existing age/count/evidence rules. LLM suggestions are allowed later, automatic anchoring is not the first step.

### `persona_state`

Current relationship and affect state from `persona_engine.py`.

This is not the same as user portrait. Keep it separate from `profile_fact`.

### Portrait Memory

A short cached stable-context block compiled from `profile_fact` and selected `anchor` buckets.

First version should be read-only. It should not include ordinary `permanent` buckets, and it should not duplicate `pinned` / `protected` by default.

## Work Order

### 1. Write this boundary plan

Done in this document. This gives later code changes one source of truth for scope.

### 2. Add a light Query Planner

Status: implemented as an optional Gateway planner, disabled by default.

Purpose: improve long-message recall without replacing the current recall path.

Trigger only when one of these is true:

- Direct recall has no hit.
- Direct recall confidence is low.
- The current message is clearly multi-topic.
- The query is long enough that one whole-message embedding/keyword search is likely to dilute the useful terms.

The planner calls a small LLM once and requires strict JSON:

```json
{
  "should_search": true,
  "queries": [
    {
      "query": "妈妈电话",
      "must_terms": ["妈妈", "电话"],
      "intent": "find recent related experience",
      "risk": "medium"
    }
  ],
  "too_vague": false
}
```

Rules:

- Keep 1 to 3 short queries.
- Run each short query independently through the existing recall path.
- Do not concatenate the planner queries back into one long sentence.
- Merge candidates in code.
- Multi-query hits get a score bonus.
- Generic-only hits get a penalty.
- A candidate must match at least one `must_terms` item to enter injection consideration.
- Candidates that fail `must_terms` may appear in debug only.
- All kept candidates still pass the current `RecallPolicy`, vague gate, budget, cooldown, and injection rules.

First version should not ask the planner to choose `keep_ids`. Add that only if short-query recall produces too much noise.

### 3. Add planner debug and a small real-query check

Status: implemented in Gateway injection debug.

Debug should show:

- Original query.
- Planner trigger reason.
- Planner JSON.
- Per-query candidates.
- Which candidates were suppressed by `must_terms`.
- Which candidates survived existing policy gates.

Use real examples that previously behaved differently under manual keyword search:

- `妈妈电话`
- `项目 delay 被批评 失眠`
- `团团 花瓶 耳机 回家`
- Other long mixed messages from recent use.

### 4. Optional: add Word Map Lite

Status: implemented as a derived diagnostic index plus private identity alias view. It remains non-injecting.

This borrows the useful part of Paw Memory's word map without importing its whole design.

It is a derived index, not a new memory layer.

Possible tables:

- `memory_word_nodes`
- `memory_word_edges`
- `memory_word_postings`

Possible term sources:

- bucket name
- domain
- tags
- profile kind
- existing content terms
- moment terms if available

Possible edge rules:

- Terms appearing on the same bucket or moment form co-occurrence edges.
- Stop words and overly common words are filtered.
- PMI or document-frequency caps prevent generic terms from dominating.

Use cases:

- Suggest query expansions for planner/debug.
- Surface strong local names or recurring concrete terms.
- Help explain why one candidate was found.

Non-goals:

- Do not replace `memory_edges`.
- Do not treat co-occurrence as evidence.
- Do not inject memories directly because a word edge exists.

### 5. Add read-only Portrait Memory cache

First implementation: done.

Implementation notes:

- Gateway supports `portrait_memory_enabled`, disabled by default.
- Source set is deterministic and read-only: `profile_fact` buckets plus selected `anchor` buckets.
- `pinned`, `protected`, ordinary `permanent`, resolved/digested/deprecated, and ordinary dynamic recall buckets are excluded by default.
- The cache key uses source bucket ids, source `updated_at`, source content hash, and relevant portrait config.
- Cache output is deterministic, not LLM-generated, so unchanged sources reuse the cached block without another model call.
- Portrait Memory is injected in stable system context as a separate `Portrait Memory` block, not inside `Core Memory`.
- Debug payload reports cache hit/miss, source ids, source roles, source hash, token estimate, and version.

First version source set:

- `profile_fact`
- selected `anchor`

Later source candidates:

- selected relationship/weather summaries, only if they do not duplicate `persona_state`

Excluded by default:

- `pinned`
- `protected`
- ordinary `permanent`
- recent dynamic recall

Cache key:

- source bucket ids
- source `updated_at`
- source content hash

If the source key is unchanged, reuse the cached block. Do not resummarize.

Injection:

- Put Portrait Memory in stable system context.
- Keep it separate from `Core Memory`.
- Make it configurable: disabled by default or guarded by a simple interval/config switch.

Debug should show:

- cache hit or miss
- source ids
- source hash
- token estimate
- generated portrait version

### 6. Add a Profile Fact page

Status: implemented as a dashboard `Profile Facts` page plus `/api/profile-facts`.

Add a small inspectable page for user portrait facts.

It should show:

- fact text
- kind
- subject / predicate / object
- evidence bucket or moment
- confidence
- last updated time
- source
- active/deprecated state

Actions:

- confirm
- edit
- deprecate
- open evidence

This page is important because user portrait should not become invisible lore.

Current shape:

- Reads existing `profile_fact` buckets and buckets with `profile_kind`.
- Shows parsed `### fact` text, kind, subject / predicate / object, evidence bucket/moment, confidence, source, update time, and active/deprecated state.
- Actions are confirm, edit, deprecate, and open evidence.
- Deprecating sets `active=false`, `deprecated=true`, `resolved=true`, and `digested=true` so stale profile facts stay out of Portrait Memory and ordinary recall.

### 7. Add semi-automatic `profile_fact` proposals

Status: implemented as manual-confirm dashboard proposals from a chosen evidence bucket.

After Portrait Memory is stable, let an LLM propose profile facts.

Proposal JSON must include:

- `fact`
- `profile_kind`
- `subject`
- `predicate`
- `object`
- `evidence_bucket_id` or `evidence_moment_id`
- `confidence`
- `reason`

Rules:

- No evidence means reject.
- First version requires manual confirmation.
- Confirmed facts use the existing `profile_fact` tool/write path.
- Rejected facts stay out of memory.

Current shape:

- Dashboard Profile Facts page accepts an evidence bucket id and optional moment id.
- `/api/profile-fact-proposals` calls the configured dehydration model and returns candidate JSON only.
- Code rejects candidates whose `evidence_bucket_id` does not match, candidates without a fact, invalid moment ids, and duplicates of existing `profile_fact` content.
- `/api/profile-fact-proposals/confirm` writes only one manually confirmed candidate through the existing `profile_fact(...)` path.

### 8. Add semi-automatic `anchor` proposals

After profile proposals behave well, allow suggestions for anchor candidates.

Rules:

- LLM can propose, not directly pin.
- Existing anchor age/count/evidence checks still apply.
- First version requires manual confirmation.
- Anchor candidates should be old or repeatedly important, not just emotionally loud today.

Current shape:

- Dashboard Profile Facts page has a separate Anchor candidate panel.
- `/api/anchor-proposals` calls the configured dehydration model and returns at most one candidate for the given bucket.
- Code rejects candidates whose `bucket_id` does not match, candidates without a reason, profile_fact buckets, feel buckets, and pinned/protected buckets.
- Existing `_can_mark_anchor()` age/count rules are checked before model generation and again through `trace(anchor=1)` on confirm.
- `/api/anchor-proposals/confirm` writes only one manually confirmed candidate through the existing `trace(bucket_id, anchor=1)` path.

### 9. Optional: candidate filter model

If planner recall returns too much noise, add a second lightweight call that sees top candidates and returns:

```json
{
  "keep_ids": ["bucket_id_1", "bucket_id_2"],
  "drop_ids": ["bucket_id_3"],
  "reason": "short explanation for debug"
}
```

Rules:

- It cannot bypass existing policy gates.
- It cannot write memory.
- It cannot inject candidates that the code rejected.

### 10. Optional: internal bucket-id detail recall

The current Gateway already injects short memory summaries with `bucket_id`.
This makes a light two-step recall possible without formal tool calling.

Status on 2026-06-05: implemented as an optional Gateway retry for non-streaming
OpenAI-compatible and Anthropic-compatible requests. It is disabled by default.
Streaming replies do not use this path because the first tokens may already have
been sent to the client.

Shape:

1. First pass injects short summaries and `bucket_id`.
2. If the model sees a relevant summary but needs details, it may put an internal request at the start of its draft:

```text
[memory_detail ids="bucket_id_1,bucket_id_2"]
```

3. Gateway intercepts this line before it reaches the user.
4. Gateway only accepts ids that were already injected in the current turn.
5. Gateway fetches the full bucket or a longer bucket detail block and asks the upstream model again with that temporary context.
6. The final user-visible reply must not contain the internal request.

Why this is preferable to a visible `[recall]` suffix on every user message:

- It does not pollute user messages or chat history.
- It does not make recall feel like part of the user's current wording.
- It avoids always showing a recall instruction to the model.
- It is lighter than formal tool calling because there is no tool schema in every request.
- It works only after Gateway has already found plausible bucket ids.

Guardrails:

- Only one internal detail recall retry per user turn.
- Limit to 2 or 3 bucket ids.
- Do not accept guessed ids.
- Do not write memory.
- Do not store the internal request in conversation history.
- Do not bypass existing recall gates; this only expands details for memories already admitted this turn.

## First Implementation Slice

The first code slice should be:

1. Query Planner config.
2. Planner prompt and strict JSON parser.
3. Trigger only on low-confidence/no-hit/multi-topic long query.
4. Run 1 to 3 short recall queries independently.
5. Merge, score, and gate with existing policy.
6. Add debug output.

Do not implement the candidate filter model in this first slice.

## Later Implementation Slice

After Query Planner has real-query evidence:

1. Add Portrait Memory read-only cache.
2. Add Profile Fact page.
3. Add manual-confirm profile fact proposals.
4. Add manual-confirm anchor proposals.
5. Use Word Map Lite diagnostics if planner debug shows repeated term-expansion misses.

Items 1 to 5 are now implemented; the candidate filter model remains optional.

## Guardrails

- No automatic `pinned`.
- No automatic `protected`.
- No automatic Core Memory edits.
- No profile fact without evidence.
- No anchor without existing gate checks.
- No planner result bypasses `RecallPolicy`.
- No generic-term-only injection.
- No fixed `[recall]` instruction appended to user messages.
- No internal memory detail request exposed to users or written into history.
- No full L0/L1/L2/L3 memory rewrite.

The guiding rule: make the existing memory more legible and better at finding what is already there, before letting it write more about the user.
