from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    1: "config/btc_1h.yaml",
    6: "config/btc_6h.yaml",
    24: "config/btc_24h.yaml",
}


def main():
    ap = argparse.ArgumentParser(description="Run reproducible BTC experiments across forecast horizons.")
    ap.add_argument("--stage", choices=["validate", "baselines", "deep", "all"], default="validate")
    ap.add_argument("--horizons", nargs="+", type=int, choices=[1, 6, 24], default=[1, 6, 24])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    for h in args.horizons:
        cfg = root / CONFIGS[h]
        print("\n" + "=" * 78)
        print(f"BTC TARGET HORIZON: {h}h | STAGE: {args.stage}")
        print("=" * 78)
        cmd = [sys.executable, str(root / "run_pipeline.py"), "--config", str(cfg), "--stage", args.stage]
        subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
