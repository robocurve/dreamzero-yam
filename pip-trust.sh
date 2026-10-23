#!/usr/bin/env bash
# Source after selecting the target interpreter; no credentials/config are read.
# Native roots must already have been refreshed by bootstrap/apt.
configure_pip_trust() {
 local interpreter="${1:?python interpreter}"
 local modern
 test -s /etc/ssl/certs/ca-certificates.crt || return 1
 modern="$("$interpreter" -I -c 'import re; from importlib.metadata import version; v=version("pip"); m=re.fullmatch(r"(\d+)\.(\d+)(?:[.a-zA-Z0-9+-]*)", v); assert m, "Unrecognized pip version"; print(int(tuple(map(int, m.groups())) >= (24, 2)))')" || return 1
 export PIP_CERT=/etc/ssl/certs/ca-certificates.crt PIP_CONFIG_FILE=/dev/null
 # Pre24.2 truststore is opt-in; remove any inherited opt-in configuration.
 export PIP_USE_FEATURE= PIP_USE_DEPRECATED=
 case "$modern" in
  1) export PIP_USE_DEPRECATED=legacy-certs ;;
  0) ;;
  *) return 1 ;;
 esac
 "$interpreter" -m pip install --only-binary=:all: --no-deps 'pip==26.2.2' || return 1
 # Subsequent installs retain the exclusive explicit bundle even after upgrades.
 export PIP_USE_DEPRECATED=legacy-certs
}
