"""Run tests against this checkout, independent of installed distributions."""

from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "tests")]
if len(sys.argv) > 1:
    unittest.main(module=None)
else:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1)
