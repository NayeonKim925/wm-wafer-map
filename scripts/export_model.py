#!/usr/bin/env python3
"""Export a trained checkpoint to a self-contained TorchScript model.

The resulting ``.pt`` loads with ``torch.jit.load`` and requires none of this
codebase, so it can be dropped into a serving runtime. A JSON sidecar records
the preprocessing contract (input channels, image size, class order).

    python scripts/export_model.py --checkpoint <ckpt> --output outputs/export/model.ts.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocessing import num_input_channels  # noqa: E402
from src.inference import Predictor  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to TorchScript for deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--output", default="outputs/export/model.ts.pt", help="Output TorchScript path."
    )
    parser.add_argument("--device", default="cpu", help="Device to export on.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = get_logger("export")

    try:
        predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    out = Path(args.output)
    predictor.export_torchscript(out)

    # Sidecar describing the preprocessing contract for the serving side.
    metadata = {
        "classes": predictor.label_mapping.to_list(),
        "image_size": predictor.image_size,
        "representation": predictor.representation,
        "input_channels": num_input_channels(predictor.representation),
        "input_shape": [
            1,
            num_input_channels(predictor.representation),
            predictor.image_size,
            predictor.image_size,
        ],
    }
    sidecar = out.with_suffix(out.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nExported TorchScript model: {out}")
    print(f"Metadata sidecar:           {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
