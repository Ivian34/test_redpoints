import argparse
import json
import sys
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _print_response(resp) -> int:
    body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


def _handle_http_error(exc: error.HTTPError) -> int:
    details = exc.read().decode("utf-8", errors="replace")
    print(f"HTTP {exc.code}: {details}", file=sys.stderr)
    return 1


def call_analyze(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        print("Error: --top-k must be > 0", file=sys.stderr)
        return 2

    payload = {
        "title": args.title,
        "top_k": args.top_k,
    }

    req = request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as resp:
            return _print_response(resp)
    except error.HTTPError as exc:
        return _handle_http_error(exc)
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1


def call_by_threshold(args: argparse.Namespace) -> int:
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    if args.stage not in (1, 2):
        print("Error: --stage must be 1 or 2", file=sys.stderr)
        return 2

    params = parse.urlencode({"threshold": args.threshold, "stage": args.stage})
    url = f"{args.url}?{params}"
    req = request.Request(url, method="GET")

    try:
        with request.urlopen(req, timeout=15) as resp:
            return _print_response(resp)
    except error.HTTPError as exc:
        return _handle_http_error(exc)
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1


def call_last_n(args: argparse.Namespace) -> int:
    if not 1 <= args.n <= 50:
        print("Error: --n must be between 1 and 50", file=sys.stderr)
        return 2

    params = parse.urlencode({"n": args.n})
    url = f"{args.url}?{params}"
    req = request.Request(url, method="GET")

    try:
        with request.urlopen(req, timeout=15) as resp:
            return _print_response(resp)
    except error.HTTPError as exc:
        return _handle_http_error(exc)
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1


def call_model_metadata(args: argparse.Namespace) -> int:
    if args.stage not in (1, 2):
        print("Error: --stage must be 1 or 2", file=sys.stderr)
        return 2

    params = parse.urlencode({"stage": args.stage})
    url = f"{args.url}?{params}"
    req = request.Request(url, method="GET")

    try:
        with request.urlopen(req, timeout=15) as resp:
            return _print_response(resp)
    except error.HTTPError as exc:
        return _handle_http_error(exc)
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call listing pipeline API endpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Call POST /analyze with a listing title"
    )
    analyze_parser.add_argument("--title", required=True, help="Listing title to analyze")
    analyze_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar references to request (default: 3)",
    )
    analyze_parser.add_argument(
        "--url",
        default=f"{DEFAULT_BASE_URL}/analyze",
        help=f"Endpoint URL (default: {DEFAULT_BASE_URL}/analyze)",
    )
    analyze_parser.set_defaults(func=call_analyze)

    threshold_parser = subparsers.add_parser(
        "by-threshold",
        help="Call GET /analyzed-listings/by-threshold",
    )
    threshold_parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help="Stage to query (1 or 2)",
    )
    threshold_parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Score threshold between 0.0 and 1.0",
    )
    threshold_parser.add_argument(
        "--url",
        default=f"{DEFAULT_BASE_URL}/analyzed-listings/by-threshold",
        help=(
            "Endpoint URL "
            f"(default: {DEFAULT_BASE_URL}/analyzed-listings/by-threshold)"
        ),
    )
    threshold_parser.set_defaults(func=call_by_threshold)

    last_n_parser = subparsers.add_parser(
        "last-n-listings",
        help="Call GET /analyzed-listings/lastN",
    )
    last_n_parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of most recent analyzed listings to return (1 to 50)",
    )
    last_n_parser.add_argument(
        "--url",
        default=f"{DEFAULT_BASE_URL}/analyzed-listings/lastN",
        help=(
            "Endpoint URL "
            f"(default: {DEFAULT_BASE_URL}/analyzed-listings/lastN)"
        ),
    )
    last_n_parser.set_defaults(func=call_last_n)

    metadata_parser = subparsers.add_parser(
        "model-metadata",
        help="Call GET /model-metadata",
    )
    metadata_parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help="Stage metadata to query (1 or 2)",
    )
    metadata_parser.add_argument(
        "--url",
        default=f"{DEFAULT_BASE_URL}/model-metadata",
        help=f"Endpoint URL (default: {DEFAULT_BASE_URL}/model-metadata)",
    )
    metadata_parser.set_defaults(func=call_model_metadata)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
