"""Central paths for the local minute-level factor research dataset."""
import os
from pathlib import Path


DATA_ROOT = Path(
    os.environ.get("MIN_GP_DATA_ROOT", r"F:\fangzheng research\data")
).expanduser()

RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
FACTOR_DIR = DATA_ROOT / "factors"
DRIPPING_STONE_DIR = FACTOR_DIR / "dripping_stone_zz500"
DRIPPING_STONE_STAGE1_DIR = DRIPPING_STONE_DIR / "stage1"
DRIPPING_STONE_VARIANTS_DIR = DRIPPING_STONE_DIR / "variants"
LIQUIDITY_ELASTICITY_DIR = FACTOR_DIR / "liquidity_elasticity_zz500"
RUSHING_FORWARD_DIR = FACTOR_DIR / "rushing_forward_zz500"
VOLUME_ENTROPY_DIR = FACTOR_DIR / "volume_entropy_zz500"
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(
    os.environ.get("MIN_GP_OUTPUT_DIR", str(PACKAGE_DIR / "output"))
).expanduser()


def output_path(name):
    """Return an output file path and create its parent directory on demand."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name

# Core inputs used by mining/evaluation. The default is the smaller CSI 500
# file because the configured PIT universe is CSI 500 and the local GPU has
# 8 GiB VRAM; the CSI 1000 source remains available for broader experiments.
MINUTE_PARQUET = RAW_DIR / "stock_zz500.parquet"
ZZ500_MINUTE_PARQUET = MINUTE_PARQUET
ZZ1000_MINUTE_PARQUET = RAW_DIR / "stock_zz1000.parquet"
ZZ500_PIT_PARQUET = RAW_DIR / "zz500_pit_daily.parquet"
ZZ1000_PIT_PARQUET = RAW_DIR / "zz1000_pit_daily.parquet"

# Daily/reference datasets available under the new data root.
DAILY_PRICE_PARQUET = INTERIM_DIR / "zz500_daily_price.parquet"
ADJUSTED_CLOSE_PARQUET = INTERIM_DIR / "zz500_adjusted_close.parquet"
INDEX_DAILY_PARQUET = INTERIM_DIR / "zz500_index_daily.parquet"
INDUSTRY_HISTORY_PARQUET = (
    RAW_DIR / "akshare_industry_value" / "sw_industry_history.parquet"
)
RISK_EXPOSURES_PARQUET = INTERIM_DIR / "zz500_risk_exposures.parquet"
INDUSTRY_VALUE_EXPOSURES_PARQUET = (
    INTERIM_DIR / "zz500_industry_value_exposures.parquet"
)
INDUSTRY_PARQUET = INDUSTRY_VALUE_EXPOSURES_PARQUET
MARKET_CAP_PARQUET = INDUSTRY_VALUE_EXPOSURES_PARQUET


def require_path(path, label):
    """Return a string path or raise a clear configuration error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            "Set MIN_GP_DATA_ROOT to override the data root."
        )
    return str(path)
