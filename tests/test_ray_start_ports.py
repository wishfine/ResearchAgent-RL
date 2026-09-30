from pathlib import Path
import os
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def configure(**ports):
    return subprocess.run(
        ["bash", "-c", 'source scripts/ray_start_ports.sh; '
         'build_ray_start_port_args || exit $?; printf "%s\\n" "${RAY_START_PORT_ARGS[@]}"'],
        cwd=ROOT, env={**os.environ, "RAY_PORT": "6400", "RAY_DASHBOARD_PORT": "8300",
                       "RAY_DASHBOARD_AGENT_PORT": "52366", **ports},
        capture_output=True, text=True,
    )


def test_all_three_ray_service_ports_are_forwarded():
    result = configure()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "--port", "6400", "--dashboard-port", "8300",
        "--dashboard-agent-listen-port", "52366",
    ]


@pytest.mark.parametrize("value", ["bad", "0", "65536", "52366; true"])
def test_invalid_agent_port_is_rejected(value):
    result = configure(RAY_DASHBOARD_AGENT_PORT=value)
    assert result.returncode != 0
    assert "RAY_DASHBOARD_AGENT_PORT" in result.stderr


def test_duplicate_agent_and_dashboard_ports_are_rejected():
    result = configure(RAY_DASHBOARD_AGENT_PORT="8300")
    assert result.returncode != 0
    assert "distinct" in result.stderr


def test_common_launcher_uses_port_args_and_waits_for_agent():
    source = (ROOT / "scripts/run_vime_hotpotqa_smoke6.sh").read_text()
    start = source.index('"$TRAIN_ENV/bin/ray" start --head')
    end = source.index("ray_started=1", start)
    assert '"${RAY_START_PORT_ARGS[@]}"' in source[start:end]
    assert "/api/local_raylet_healthz" in source
