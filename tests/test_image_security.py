"""CPU-only checks; do not import Modal apps or construct cloud resources."""

import ast
import importlib.util
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_security", ROOT / "src/image_security.py")
security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security)


class RecordingImage:
    def __init__(self):
        self.steps = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.steps.append((name, args, kwargs))
            return self
        return record


class ImageSecurityTest(unittest.TestCase):
    def test_native_roots_and_constraint_exist_before_first_pip_operation(self):
        image = RecordingImage()
        self.assertIs(security.secure_image(image), image)
        self.assertEqual([step[0] for step in image.steps],
                         ["apt_install", "add_local_file", "add_local_file", "env",
                          "run_commands", "env", "pip_install"])
        self.assertIn("ca-certificates", image.steps[0][1])
        local, remote = image.steps[1][1]
        self.assertEqual(Path(local).resolve(), security.SECURITY_CONSTRAINTS.resolve())
        self.assertTrue(image.steps[1][2]["copy"])
        self.assertEqual(image.steps[2][1], (str(security.PIP_BOOTSTRAP), security.IMAGE_BOOTSTRAP))
        env = image.steps[3][1][0]
        self.assertEqual(env["PIP_CONSTRAINT"], remote)
        self.assertEqual(env["PIP_CERT"], "/etc/ssl/certs/ca-certificates.crt")
        self.assertIn("configure_pip_trust python", image.steps[4][1][0])
        self.assertEqual(image.steps[5][1][0]["PIP_USE_DEPRECATED"], "legacy-certs")
        self.assertEqual(image.steps[6][1], ("certifi>=2026.10.19",))

    def test_every_existing_image_is_wrapped(self):
        images = []
        for path in (ROOT / "src").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "modal"
                        and node.func.value.attr == "Image"):
                    parents = [parent for parent in ast.walk(tree)
                               if node in ast.iter_child_nodes(parent)]
                    self.assertEqual(len(parents), 1)
                    parent = parents[0]
                    self.assertIsInstance(parent, ast.Call)
                    self.assertIsInstance(parent.func, ast.Name)
                    self.assertEqual(parent.func.id, "secure_image", str(path))
                    images.append((path.name, node.func.attr))
        self.assertEqual(len(images), 5)

    def test_persistent_floor_and_local_lock(self):
        constraints = [line for line in security.SECURITY_CONSTRAINTS.read_text().splitlines()
                       if line and not line.startswith("#")]
        self.assertEqual(constraints, ["certifi>=2026.10.19", "aiohttp>=3.14.4"])
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertIn(constraints[0], manifest["project"]["dependencies"])
        self.assertIn("aiohttp>=3.14.4", manifest["project"]["dependencies"])
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        certifi = next(package for package in lock["package"] if package["name"] == "certifi")
        self.assertGreaterEqual(tuple(map(int, certifi["version"].split("."))), (2026, 10, 19))
        aiohttp = next(package for package in lock["package"] if package["name"] == "aiohttp")
        self.assertGreaterEqual(tuple(map(int, aiohttp["version"].split("."))), (3, 14, 4))


if __name__ == "__main__":
    unittest.main()
