import logging
import sys
import yaml 
import os

from dsl.generators.booking import generate_booking_df
from dsl.preprocessing.features import build_booking_features
from dsl.preprocessing.leakage import leakage_audit
from dsl.preprocessing.splits import temporal_split
#from src.dsl.run_expirement import run_expirement
from dsl.tracking.registry import log_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/booking_clf.yaml"
cfg = yaml.safe_load(open(cfg_path))

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# generate the data
logger.info("step0: generate booking data")
df, schema = generate_booking_df( 
    n_samples=cfg["data"]["n_samples"],
    seed=cfg["data"]["seed"]
)

logger.info("step1: feature engineering - adding features")
df, schema = build_booking_features(df, schema)

logger.info("step1: feature engineering - data leakage")
schema = leakage_audit(df, schema)

used_cols = (schema["num"] + schema["cat"] + schema["ids"] + [schema["target"], schema["timestamp"]])
df = df[used_cols]

logger.info("step2: split data")
splits = temporal_split(df, schema, schema["timestamp"], .7, .85)


logger.info("step3: run the ML training experiment")

for model_name in ["lgbm", "lgr"]:
    run_cfg = {
        **cfg,
        "model": {"name": model_name, "params": cfg["model"]["params"] if model_name == cfg["model"]["name"] else {}},
        "experiment": {**cfg["experiment"], "name": f"baseline_{model_name}"},
    }
    run_id = log_experiment(run_cfg, splits, schema, tracking_uri)

# test the serving.
#import json
#json.dump(splits.Xva.head(10).to_dict(orient="records"), open("payload.json", "w"))