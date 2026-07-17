#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ops_common.sh"

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

TEST_CONTAINER_ID="container-id"
TEST_BIND_SOURCE=""

ombre_compose() {
  [[ -n "${TEST_CONTAINER_ID}" ]] && printf '%s\n' "${TEST_CONTAINER_ID}"
}

docker() {
  printf '%s\n' "${TEST_BIND_SOURCE}"
}

fail() {
  printf 'FAIL %s\n' "${1}" >&2
  exit 1
}

regular_file="${TEST_ROOT}/config.yaml"
printf 'identity: {}\n' > "${regular_file}"
TEST_BIND_SOURCE="${regular_file}"
ombre_validate_compose_file_bind compose.yml ombre-brain /app/config.yaml config.yaml >/dev/null

missing_file="${TEST_ROOT}/missing/config.yaml"
TEST_BIND_SOURCE="${missing_file}"
if ombre_validate_compose_file_bind compose.yml ombre-brain /app/config.yaml config.yaml >"${TEST_ROOT}/missing.out" 2>"${TEST_ROOT}/missing.err"; then
  fail 'missing bind source was accepted'
fi
[[ ! -e "${missing_file}" ]] || fail 'missing bind source was created'
grep -Fq '已停止部署' "${TEST_ROOT}/missing.err" || fail 'missing-source error did not explain the stop'

directory_source="${TEST_ROOT}/directory.yaml"
mkdir "${directory_source}"
TEST_BIND_SOURCE="${directory_source}"
if ombre_validate_compose_file_bind compose.yml ombre-brain /app/config.yaml config.yaml >"${TEST_ROOT}/directory.out" 2>"${TEST_ROOT}/directory.err"; then
  fail 'directory bind source was accepted'
fi
grep -Fq '不是普通文件' "${TEST_ROOT}/directory.err" || fail 'directory-source error was unclear'

TEST_CONTAINER_ID=""
fallback_compose="${TEST_ROOT}/compose.yml"
printf '%s\n' \
  'services:' \
  '  ombre-brain:' \
  '    volumes:' \
  '      - ./config.yaml:/app/config.yaml:ro' > "${fallback_compose}"
fallback_source="$(ombre_compose_bind_source "${fallback_compose}" ombre-brain /app/config.yaml)"
[[ "${fallback_source}" == "${regular_file}" ]] || fail 'compose-file fallback resolved the wrong source'

guard_line="$(grep -n 'ombre_validate_compose_file_bind' "${SCRIPT_DIR}/update_deploy.sh" | head -n 1 | cut -d: -f1)"
update_line="$(grep -n 'Update containers' "${SCRIPT_DIR}/update_deploy.sh" | head -n 1 | cut -d: -f1)"
(( guard_line < update_line )) || fail 'bind guard must run before docker compose up'

export OMBRE_ONE_CLICK_SOURCE_ONLY=1
source "${SCRIPT_DIR}/one_click.sh"
actual_config="${TEST_ROOT}/mounted-config.yaml"
actual_env="${TEST_ROOT}/mounted.env"
backup_compose="${TEST_ROOT}/backup-compose.yml"
printf 'identity: {ai_name: Haven}\n' > "${actual_config}"
printf 'OMBRE_API_KEY=test-only\n' > "${actual_env}"
printf 'services: {}\n' > "${backup_compose}"
COMPOSE_FILE="${backup_compose}"
DEPLOY_TARGET="vps"

run_target_shell() {
  return 0
}

ombre_compose_bind_source() {
  case "${3}" in
    /app/config.yaml) printf '%s\n' "${actual_config}" ;;
    /app/.env) printf '%s\n' "${actual_env}" ;;
    *) return 1 ;;
  esac
}

backup_current_deployment test_bind_source >"${TEST_ROOT}/backup.out"
config_backup="$(find "${TEST_ROOT}" -maxdepth 1 -type f -name 'mounted-config.yaml.bak.*' -print -quit)"
env_backup="$(find "${TEST_ROOT}" -maxdepth 1 -type f -name 'mounted.env.bak.*' -print -quit)"
[[ -n "${config_backup}" ]] || fail 'actual config bind source was not backed up'
[[ -n "${env_backup}" ]] || fail 'actual env bind source was not backed up'
cmp -s "${actual_config}" "${config_backup}" || fail 'config bind backup content changed'
cmp -s "${actual_env}" "${env_backup}" || fail 'env bind backup content changed'

printf 'ops bind guard tests passed\n'
