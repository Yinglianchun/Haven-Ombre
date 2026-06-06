import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dehydrator import Dehydrator
from persona_engine import PersonaStateEngine


class RecordingChatClient:
    def __init__(self, content):
        self.calls = []
        self.content = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class QueueRecordingChatClient:
    def __init__(self, contents):
        self.calls = []
        self.contents = list(contents)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_dehydrator_sends_disabled_thinking_mode_when_configured(test_config):
    cfg = deepcopy(test_config)
    cfg["dehydration"].update(
        api_key="test-key",
        model="deepseek-v4-flash",
        thinking_mode="disabled",
    )
    dehydrator = Dehydrator(cfg)
    client = RecordingChatClient("short summary")
    dehydrator.client = client

    result = await dehydrator._api_dehydrate("这是一段需要脱水的长文本")

    assert result == "short summary"
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_dehydrator_omits_thinking_mode_by_default(test_config):
    cfg = deepcopy(test_config)
    cfg["dehydration"].update(
        api_key="test-key",
        model="deepseek-v4-flash",
    )
    dehydrator = Dehydrator(cfg)
    client = RecordingChatClient("short summary")
    dehydrator.client = client

    await dehydrator._api_dehydrate("这是一段需要脱水的长文本")

    assert "extra_body" not in client.calls[0]


@pytest.mark.asyncio
async def test_dehydrator_direct_capsule_uses_separate_cache(test_config):
    cfg = deepcopy(test_config)
    cfg["dehydration"].update(api_key="test-key")
    dehydrator = Dehydrator(cfg)
    client = QueueRecordingChatClient(["normal summary", "direct capsule"])
    dehydrator.client = client
    content = "long bucket content " * 140

    normal = await dehydrator.dehydrate(content)
    capsule = await dehydrator.dehydrate_direct_capsule(content)
    cached_capsule = await dehydrator.dehydrate_direct_capsule(content)
    cached_normal = await dehydrator.dehydrate(content)

    assert normal == "normal summary"
    assert capsule == "direct capsule"
    assert cached_capsule == "direct capsule"
    assert cached_normal == "normal summary"
    assert len(client.calls) == 2
    assert client.calls[0]["messages"][0]["content"] != client.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_persona_sends_enabled_thinking_mode_when_configured(test_config):
    cfg = deepcopy(test_config)
    cfg["dehydration"]["api_key"] = ""
    cfg["persona"].update(
        api_key="test-key",
        model="deepseek-v4-flash",
        thinking_mode="enabled",
    )
    engine = PersonaStateEngine(cfg)
    client = RecordingChatClient(
        json.dumps(
            {
                "event_type": "neutral",
                "perceived_intent": "user says hi",
                "affect_delta": {"valence": 0.01, "arousal": 0.0},
                "relationship_event": False,
                "relationship_delta": {
                    "affinity": 0.0,
                    "dominance": 0.0,
                    "defensiveness": 0.0,
                    "trust": 0.0,
                },
                "personality_signal": False,
                "personality_delta": {
                    "openness": 0.0,
                    "conscientiousness": 0.0,
                    "extraversion": 0.0,
                    "agreeableness": 0.0,
                    "neuroticism": 0.0,
                },
                "mood_label": "warm_neutral",
                "residue": "",
                "confidence": 0.8,
            },
            ensure_ascii=False,
        )
    )
    engine.client = client

    await engine.update_from_exchange("sess-thinking", "哥哥在吗", "在。")

    assert client.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
