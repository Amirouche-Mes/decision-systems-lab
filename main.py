from src.decision_systems_lab.functions import generate_booking_df, data_cont_leakage_detection, compute_iv_auc, make_split
import pandas as pd

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
    time_sice_last_click_pd=(
        pl.col("click_ts") - pl.col("click_ts").shift(1).over("partner_id", order_by="click_ts")
    ).dt.total_seconds().fill_null(-1),
    clicks_last_1h_pd = pl.col("click_ts").count().over("partner_id", order_by="click_ts").fill_null(-1),
    time_since_last_click_pd=(
        pl.col("click_ts") - pl.col("click_ts").shift(1).over("geo", order_by="click_ts")
    ).dt.total_seconds().fill_null(-1),
    clicks_last_1h_geo = pl.col("click_ts").count().over("geo", order_by="click_ts").fill_null(-1)
)

df_booking = df_pl.to_pandas()

# final step cleaning -> 
df_booking = df_booking.drop(columns=["minutes_on_ota_site", "booking_amount"])

# let's make the split 
train, valid, test = make_split(df_booking, "click_ts", .7, .85)

# trainlightgbm 
import lightgbm as lgb 

FEATS = ['partner_id', 'device', 'geo', 'price', 'lead_time_days',
         'nights', 'past_click_30d', 'converted', 'time_sice_last_click_pd',
         'clicks_last_1h_pd', 'time_since_last_click_pd', 'clicks_last_1h_geo']
CAT = ["partener_id", "device","geo"]

def prep(d, cats=None):
    d = d[FEATS].copy()
    for c in CAT:
        d[c] = d[c].astype("category") if cats is None else pd.Categorical(d[c], categories=cats[c])
    return d 
Xtr = prep(train); cats = {c: Xtr[c].cat.categories for c in CAT}
Xva, Xte = prep(valid, cats), prep(test, cats)

# let's make the first training
model = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, 
                           min_child_samples=50, random_state=0, verbose=-1,
                           class_weight="balanced")

# 
from sklearn.linear_model import LogisticRegression

model_lgr = LogisticRegression(penalty="l2", C="1.0", solver="lbfgs", 
                               max_iter=1000, random_state=42,)



