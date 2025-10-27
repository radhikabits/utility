import sys
from pathlib import Path

# Ensure src layout is discoverable for tests without installation
root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
