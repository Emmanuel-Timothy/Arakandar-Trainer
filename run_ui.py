"""Launch the purpose-built Streamlit analyst workspace."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "app" / "ui.py")])


if __name__ == "__main__":
    main()
