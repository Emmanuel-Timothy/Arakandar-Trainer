"""Run the authenticated Arakandar HTTP service."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.api:app",
        host=os.getenv("ARAKANDAR_HOST", "127.0.0.1"),
        port=int(os.getenv("ARAKANDAR_PORT", "8000")),
        reload=os.getenv("ARAKANDAR_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
