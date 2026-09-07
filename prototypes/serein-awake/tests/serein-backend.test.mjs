import test from 'node:test';
import assert from 'node:assert/strict';
import { callSereinBackend, sereinConfigured } from '../server/sereinBackend.mjs';

test('server transport preserves save conflict and never falls back to the old backend', async () => {
  const env = { SEREIN_MEMORY_URL: 'http://127.0.0.1:18105/', SEREIN_MEMORY_TOKEN: 'fixture-token' };
  assert.equal(sereinConfigured(env), true);
  const calls = [];
  const fetchImpl = async (url, request) => {
    calls.push({ url, request });
    return { ok: false, status: 409, json: async () => ({ status: 'conflict', reason: 'revision_mismatch' }) };
  };
  const body = { narrative_id: 'narrative_test', expected_revision: 1 };
  const result = await callSereinBackend('/api/narrative-rolls/save-body', { method: 'POST', body }, { env, fetchImpl });
  assert.equal(result.status, 409);
  assert.equal(result.payload.reason, 'revision_mismatch');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'http://127.0.0.1:18105/api/narrative-rolls/save-body');
  assert.equal(calls[0].request.headers.Authorization, 'Bearer fixture-token');
  assert.deepEqual(JSON.parse(calls[0].request.body), body);
});

test('missing new backend credential does not reuse an old credential or call a server', async () => {
  const env = { SEREIN_MEMORY_URL: 'http://127.0.0.1:18105', OMBRE_GATEWAY_TOKEN: 'old-token' };
  let called = false;
  await assert.rejects(callSereinBackend('/api/handoff-scenes', {}, { env, fetchImpl: async () => { called = true; } }),
    /serein_backend_not_configured/);
  assert.equal(called, false);
});
