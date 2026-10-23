"""No network/package execution: record the bootstrap subprocess environment."""

import json
import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "pip-trust.sh"


@pytest.mark.parametrize("mode,initial", [("0", ""), ("1", "legacy-certs")])
def test_exclusive_bundle_before_bootstrap_and_after_pip_upgrade(tmp_path, mode, initial):
    interpreter = tmp_path / "python"
    record = tmp_path / "record.json"
    interpreter.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-I': print(os.environ['FIXTURE_MODE'])\n"
        "else:\n"
        " with open(os.environ['FIXTURE_RECORD'], 'w') as f:\n"
        "  json.dump({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in "
        "['PIP_CERT', 'PIP_CONFIG_FILE', 'PIP_USE_FEATURE', 'PIP_USE_DEPRECATED']}}, f)\n"
        " sys.exit(int(os.environ.get('FIXTURE_FAIL', '0')))\n"
    )
    interpreter.chmod(0o700)
    env = dict(os.environ, FIXTURE_MODE=mode, FIXTURE_RECORD=str(record),
               PIP_USE_FEATURE="truststore", PIP_USE_DEPRECATED="other")
    command = '. "$1"; configure_pip_trust "$2"; printf "%s" "$PIP_USE_DEPRECATED"'
    completed = subprocess.run(["bash", "-e", "-c", command, "test", str(HELPER),
                                str(interpreter)], env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "legacy-certs"
    observed = json.loads(record.read_text())
    assert observed["argv"] == ["-m", "pip", "install", "--only-binary=:all:",
                                 "--no-deps", "pip==26.2.2"]
    assert observed["env"] == {
        "PIP_CERT": "/etc/ssl/certs/ca-certificates.crt", "PIP_CONFIG_FILE": "/dev/null",
        "PIP_USE_FEATURE": "", "PIP_USE_DEPRECATED": initial,
    }
    # A failed bootstrap must prevent subsequent package installation.
    failed = subprocess.run(["bash", "-e", "-c", command, "test", str(HELPER),
                             str(interpreter)], env=dict(env, FIXTURE_FAIL="1"),
                            capture_output=True, text=True)
    assert failed.returncode != 0


def test_unknown_installer_mode_fails_before_package_execution(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/sh\n[ "$1" = -I ] && echo unknown\n')
    interpreter.chmod(0o700)
    completed = subprocess.run(["bash", "-e", "-c", '. "$1"; configure_pip_trust "$2"',
                                "test", str(HELPER), str(interpreter)], capture_output=True)
    assert completed.returncode != 0
