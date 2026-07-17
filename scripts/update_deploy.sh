#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ops_common.sh"

cd "$(ombre_repo_root)"

COMPOSE_FILE="$(ombre_compose_file)"
HEALTH_URL="${HEALTH_URL:-$(ombre_default_health_url "${COMPOSE_FILE}")}"
GATEWAY_SERVICE="${GATEWAY_SERVICE:-${OMBRE_GATEWAY_SERVICE:-ombre-gateway}}"

echo "Repo: $(pwd)"
echo "Compose: ${COMPOSE_FILE}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ombre_update_git_checkout
fi

BRAIN_SERVICE="${OMBRE_SERVICE:-ombre-brain}"
ombre_validate_compose_file_bind "${COMPOSE_FILE}" "${BRAIN_SERVICE}" "/app/config.yaml" "config.yaml"

echo "Update containers..."
if grep -Eq '^[[:space:]]*build:' "${COMPOSE_FILE}"; then
  ombre_compose -f "${COMPOSE_FILE}" up -d --build --remove-orphans
else
  ombre_compose -f "${COMPOSE_FILE}" pull
  ombre_compose -f "${COMPOSE_FILE}" up -d --remove-orphans
fi

ombre_compose -f "${COMPOSE_FILE}" ps
ombre_wait_for_health "${HEALTH_URL}" "${HEALTH_TRIES:-30}" "${HEALTH_DELAY:-2}"
if ombre_compose_service_exists "${COMPOSE_FILE}" "${GATEWAY_SERVICE}"; then
  GATEWAY_HEALTH_URL="${GATEWAY_HEALTH_URL:-$(ombre_compose_service_health_url "${COMPOSE_FILE}" "${GATEWAY_SERVICE}" "8010" "http://127.0.0.1:18002/health")}"
  ombre_wait_for_health "${GATEWAY_HEALTH_URL}" "${HEALTH_TRIES:-30}" "${HEALTH_DELAY:-2}"
else
  echo "Gateway service not found in compose; skip gateway health check."
fi

echo "Update done."
