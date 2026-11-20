"""Start the local demo without exposing it to the network."""

import argparse
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", type=Path, default=Path(".data/claims-demo"))
    parser.add_argument("--data", type=Path, default=Path(".data"))
    args = parser.parse_args()
    load_dotenv(override=False)
    try:
        import uvicorn

        from .app import create_app
    except ImportError as exc:
        raise SystemExit(
            "Install the demo dependencies first: uv sync --extra demo"
        ) from exc
    uvicorn.run(
        create_app(store_root=args.store, data_root=args.data),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
