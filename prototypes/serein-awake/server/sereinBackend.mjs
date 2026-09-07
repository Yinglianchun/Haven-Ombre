// Server-only memory backend. Credentials never enter the browser bundle.
export function sereinConfigured(env = process.env) {
  return Boolean(String(env.SEREIN_MEMORY_URL || "").trim());
}

export async function callSereinBackend(path, { method = "GET", body } = {}, { env = process.env, fetchImpl = fetch } = {}) {
  const base = String(env.SEREIN_MEMORY_URL || "").trim().replace(/\/$/, "");
  const token = String(env.SEREIN_MEMORY_TOKEN || "").trim();
  if (!base || !token) throw new Error("serein_backend_not_configured");
  const response = await fetchImpl(`${base}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/json",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),
  });
  return { ok: response.ok, status: response.status, payload: await response.json() };
}

export async function callSereinTool(name, args) {
  const result = await callSereinBackend("/v1/tools/call", { method: "POST", body: { name, arguments: args } });
  if (!result.ok) throw new Error(`serein_tool_${result.status}`);
  return result.payload.result;
}
