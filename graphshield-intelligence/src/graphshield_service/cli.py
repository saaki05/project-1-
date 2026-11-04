"""``graphshield-serve --bundle model.pt`` starts the HTTP service."""

from __future__ import annotations

import argparse

import uvicorn

from graphshield_service.api import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="graphshield-serve")
    ap.add_argument("--bundle", required=True, help="model bundle saved with save_bundle")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(args.bundle), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
