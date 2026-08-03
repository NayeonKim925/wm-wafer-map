#!/usr/bin/env python3
"""Classify wafer maps with a trained checkpoint.

    python inference.py --checkpoint <ckpt> --input wafer.npy
    python inference.py --checkpoint <ckpt> --input some_wafers.pkl --top-k 5 --output preds.json

``--input`` accepts a ``.npy`` file (a single 2-D wafer map or an object array
of them) or a ``.pkl`` file (a DataFrame with a ``waferMap`` column, or a list
of wafer maps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference import Predictor, load_wafer_maps  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained WM-811K classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--input", required=True, help="Path to a .npy or .pkl of wafer map(s).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of classes to report.")
    parser.add_argument(
        "--device", default=None, help="Force a device (cpu | cuda | mps); default auto."
    )
    parser.add_argument("--output", default=None, help="Optional path to write predictions JSON.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    parser.add_argument(
        "--limit", type=int, default=20, help="Max predictions to print to the console."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = get_logger("inference")

    try:
        predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
        wafer_maps = load_wafer_maps(args.input)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Running inference on %d wafer map(s).", len(wafer_maps))
    results = predictor.predict(wafer_maps, top_k=args.top_k)

    # User-facing CLI output.
    for i, result in enumerate(results[: args.limit]):
        top = ", ".join(f"{c['class']}={c['probability']:.3f}" for c in result["top_k"])
        print(
            f"[{i}] {result['predicted_class']} "
            f"(confidence={result['confidence']:.3f}) | top-{args.top_k}: {top}"
        )
    if len(results) > args.limit:
        print(f"... {len(results) - args.limit} more (use --limit or --output).")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {len(results)} predictions to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
