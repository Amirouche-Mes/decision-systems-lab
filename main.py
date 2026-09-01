from src.decision_systems_lab.generators import generate_booking_df
from src.decision_systems_lab.functions import data_cont_leakage_detection, compute_iv_auc, make_split
import pandas as pd
import importlib

# generate data
df_booking = generate_booking_df()

# preprocessing step
print("shape:", df_booking.shape)
print("target rate:", df_booking.converted.mean().round(4))
print("duplicates:", df_booking.duplicated().sum())
print("date range:", df_booking.click_ts.min(), "->", df_booking.click_ts.max())
print("null columns:", (df_booking.isna().mean()*100).round(2).sort_values(ascending=False).head())

df_booking = df_booking.drop_duplicates().reset_index(drop=True)
target_rate = df_booking.converted.mean().round(4)

na_columns_serie = (df_booking.isna().mean()*100).round(2).sort_values(ascending=False)
num_cols = set(df_booking.select_dtypes(include='number').columns)
cat_cols = df_booking.select_dtypes(include=["object", "str", "string", "category"]).columns.to_list()

int_col_data_leakage = data_cont_leakage_detection(num_cols, df_booking, df_booking.converted, target_rate, na_columns_serie)


cat_col_leakage_df_result = pd.Series(cat_cols, index=cat_cols).apply(
    lambda col: compute_iv_auc(df_booking, col, target_col="converted")
)

cat_col_leakage_df_result = cat_col_leakage_df_result.sort_values(by="AUC", ascending=False)

cat_col_leakage_df_result.head()

# feature enginering 

PRIOR_W = 20.0
df_booking = df_booking.sort_values("click_ts").reset_index(drop=True)
prior = df_booking.converted.mean()

# expending, features to add with more valuable information 
# this section could be replaced by the category_encoder library, but it doesn't handle the leakage of the conv variable
# possible variables to add that might help, 
# dynamique frequency: time since last click, clicks last 24h, clicks on sessions, inactivity_time, price_ratio = price / avgprice_for_geo
# price per night, frequency_encoding per partner, cyclic hours hour_sin = sin(2p*hour/24), hour_cos=...
# # we can also add the embedding o
g = df_booking.groupby("partner_id")["converted"]
df_booking["partner_rate"] = (g.cumsum() - df_booking.converted + prior * PRIOR_W)/(g.cumcount() + PRIOR_W)

import polars as pl 

df_pl = pl.from_pandas(df_booking)

df_pl = df_pl.with_columns(
    # time since the last click of the partner_id 
    time_since_last_click_pd=(
        pl.col("click_ts") - pl.col("click_ts").shift(1).over("partner_id", order_by="click_ts")
    ).dt.total_seconds().fill_null(-1),
    clicks_last_1h_pd = (pl.col("click_ts").cum_count().over("partner_id", order_by="click_ts")-1).fill_null(-1),
    time_since_last_click_geo=(
        pl.col("click_ts") - pl.col("click_ts").shift(1).over("geo", order_by="click_ts")
    ).dt.total_seconds().fill_null(-1),
    clicks_last_1h_geo = (pl.col("click_ts").cum_count().over("geo", order_by="click_ts")-1).fill_null(-1)
)

df_booking = df_pl.to_pandas()

# final step cleaning -> 
df_booking = df_booking.drop(columns=["minutes_on_ota_site", "booking_amount"])

# let's make the split 
train, valid, test = make_split(df_booking, "click_ts", .7, .85)

# trainlightgbm 
num_cols = ['price', 'lead_time_days',
            'nights', 'past_click_30d', 'time_since_last_click_pd',
            'clicks_last_1h_pd', 'time_since_last_click_geo', 'clicks_last_1h_geo', 'partner_rate']
cat_cols = ["partner_id", "device","geo"]

low_card_cols = ["geo", "device"]
high_card_cols = ["partner_id"]


Xtr, ytr = train.drop(columns=["converted"]).copy(), train.converted.copy()
Xva, yva = valid.drop(columns=["converted"]).copy(), valid.converted.copy()
Xte, yte = test.drop(columns=["converted"]).copy(), test.converted.copy()

import src.decision_systems_lab.functions as dsl
importlib.reload(dsl)
train_lgbm = dsl.train_lgbm
train_lgr = dsl.train_lgr
evaluate = dsl.evaluate
run_experiment = dsl.run_experiment

from collections import namedtuple

Splits = namedtuple("Splits", ["Xtr", "Ytr", "Xva", "yva", "Xte", "Yte"])

cols = {
    "num": num_cols, 
    "cat": cat_cols,
    "low_card": low_card_cols,
    "high_card": high_card_cols
}


lgb_model = train_lgbm(Splits, cols,)
lgr_model = train_lgr(Splits, cols,)


# get the metrics of each model 
lgb_metrics = evaluate(lgb_model, Xva, yva)
lgr_metrics = evaluate(lgr_model, Xva, yva)

# run experiment: this step should only leads us to run many experiments
results = []

results.append(
    run_experiment(
        name="lgb_baseline",
        model_name="lgb",
        Splits,
        cols,
    )
)

results.append(
    run_experiment(
            name="lgr_baseline",
            model_name="lgr",
            Splits,
            cols,
        )
)

df_results = pd.DataFrame(results)
print(df_results)

