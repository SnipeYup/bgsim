"""Launch the bgsim web app. Run: python web/run.py [--port 8000]

Reads a .env file at the repo root if present (KEY=value lines), so your
API key can live there instead of being typed into a terminal. .env is
gitignored — never commit it.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    import uvicorn
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print("engine generation:", "enabled" if os.environ.get("ANTHROPIC_API_KEY")
          else "disabled (no ANTHROPIC_API_KEY — built-in engines still work)")
    uvicorn.run("web.app:app", host="127.0.0.1", port=port)
