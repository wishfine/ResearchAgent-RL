#!/usr/bin/env bash
# Sourced by launchers: a dashboard HTTP port does not isolate the agent port.
build_ray_start_port_args() {
  local name value port
  local ports=()
  for name in RAY_PORT RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_PORT; do
    value="${!name}"
    if [[ ! "$value" =~ ^[0-9]{1,5}$ ]]; then
      echo "[ERROR] $name must be a port in 1..65535" >&2
      return 1
    fi
    port=$((10#$value))
    if (( port < 1 || port > 65535 )); then
      echo "[ERROR] $name must be a port in 1..65535" >&2
      return 1
    fi
    ports+=("$port")
  done
  if [[ "${ports[0]}" == "${ports[1]}" || "${ports[0]}" == "${ports[2]}" ||
        "${ports[1]}" == "${ports[2]}" ]]; then
    echo "[ERROR] Ray head, dashboard and dashboard agent ports must be distinct" >&2
    return 1
  fi
  RAY_PORT="${ports[0]}"
  RAY_DASHBOARD_PORT="${ports[1]}"
  RAY_DASHBOARD_AGENT_PORT="${ports[2]}"
  RAY_START_PORT_ARGS=(--port "$RAY_PORT" --dashboard-port "$RAY_DASHBOARD_PORT"
                       --dashboard-agent-listen-port "$RAY_DASHBOARD_AGENT_PORT")
}
