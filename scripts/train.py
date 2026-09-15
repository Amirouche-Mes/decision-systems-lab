import logging
import sys
import yaml 

from src.dsl.generators.booking import generate_booking_df
from src.dsl.preprocessing.features import build_booking_features
from src.dsl.preprocessing.leakage import leakage_audit
from src.dsl.preprocessing.splits import temporal_split
from src.dsl.run_expirement import run_expirement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

cfg = yaml.safe_load(open("configs/booking_clf.yaml"))


# generate the data
logger.info("step0: generate booking data")
df, schema = generate_booking_df( 
    n_samples=cfg["data"]["n_samples"],
    seed=cfg["data"]["seed"]
)

logger.info("step1: feature engineering - adding features")
df, schema = build_booking_features(df, schema)

logger.info("step1: feature engineering - data leakage")
clean_schema = leakage_audit(df, schema)

used_cols = (clean_schema["num"] + clean_schema["cat"] + schema["ids"] + [clean_schema["target"], schema["timestamp"]])
df = df[used_cols]

logger.info("step2: split data")
splits = temporal_split(df, clean_schema, clean_schema["timestamp"], .7, .85)

logger.info("step3: train the ML model experiment")
expr_results = run_expirement(
                name=cfg["experiment"]["name"],
                model_name=cfg["model"]["name"],
                splits=splits,
                feature_cols=clean_schema,
                model_params=cfg["model"]["params"],
                include_test=cfg["experiment"]["include_test"],
    )








