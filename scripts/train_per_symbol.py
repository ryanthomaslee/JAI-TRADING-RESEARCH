"""
Train per-symbol models: XGBoost and LightGBM.

Usage:
    uv run python -m scripts.train_per_symbol
    uv run python -m scripts.train_per_symbol --algo xgb
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.splits      import PurgedWalkForwardSplit
from src.models.trainer     import train_per_symbol
from src.models.xgboost_model  import XGBoostModel
from src.models.lightgbm_model import LightGBMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["xgb", "lgbm", "both"], default="both")
    args = parser.parse_args()

    splitter = PurgedWalkForwardSplit(n_splits=5, embargo_days=5, purge_days=20)

    if args.algo in ("xgb", "both"):
        logging.getLogger().info("Training per-symbol XGBoost…")
        train_per_symbol(
            model_cls=XGBoostModel,
            splitter=splitter,
            strategy_name="per_symbol_xgb",
        )

    if args.algo in ("lgbm", "both"):
        logging.getLogger().info("Training per-symbol LightGBM…")
        train_per_symbol(
            model_cls=LightGBMModel,
            splitter=splitter,
            strategy_name="per_symbol_lgbm",
        )

if __name__ == "__main__":
    main()
