"""Trust-store defaults shared by future Modal image rebuilds (no Modal import)."""

from pathlib import Path

SECURITY_CONSTRAINTS = Path(__file__).resolve().parents[1] / "security-constraints.txt"
IMAGE_CONSTRAINTS = "/opt/dreamzero-security-constraints.txt"
PIP_BOOTSTRAP = SECURITY_CONSTRAINTS.with_name("pip-trust.sh")
IMAGE_BOOTSTRAP = "/opt/dreamzero-pip-trust.sh"


def secure_image(image):
    """Refresh native roots before pip, then retain the certifi floor on later installs."""
    return (
        image.apt_install("ca-certificates")
        .add_local_file(str(SECURITY_CONSTRAINTS), IMAGE_CONSTRAINTS, copy=True)
        .add_local_file(str(PIP_BOOTSTRAP), IMAGE_BOOTSTRAP, copy=True)
        .env({
            "PIP_CONSTRAINT": IMAGE_CONSTRAINTS,
            "PIP_CERT": "/etc/ssl/certs/ca-certificates.crt",
        })
        .run_commands(f". {IMAGE_BOOTSTRAP} && configure_pip_trust python")
        .env({"PIP_USE_DEPRECATED": "legacy-certs", "PIP_CONFIG_FILE": "/dev/null",
              "PIP_USE_FEATURE": ""})
        .pip_install("certifi>=2026.10.19")
    )
