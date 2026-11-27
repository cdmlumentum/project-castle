"""Run this checkout even when another vlac distribution is installed."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if len(sys.argv) > 1 and sys.argv[1] == "mesh-study":
    from vlac.mesh_study import main
    raise SystemExit(main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "native":
    from vlac.native_mode import main
    raise SystemExit(main(sys.argv[2:]))

from vlac.cli import main
raise SystemExit(main(sys.argv[1:]))
