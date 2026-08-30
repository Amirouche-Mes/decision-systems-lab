# first function to generate the data
import numpy as np, pandas as pd

def generate_booking_df():
    """
        The main objective of the function is to generate a dataframe that has some caracteristics, the main goal is to built a converted response with the 
        logic followed: 
        logit : bias started at -3.4 if the other factors will be at 0, the base convertion propability will be samll, this is reflects the reality of otas
             for the 0.8 (device == 'desktop'), we simulate the observed bias in e commerce or online browsing in general, people are more willing to converte when using their laptop
             0.55 * np.log1p(past_clicks), np.log1p (compute(ln(1+x))), the more the lead clicks the more the chance to converte, but the effect of the first clic is huge comparing to 50 or 51 th one
             optimal price, -0.8 (log(price)- 5), non linear relation ship, when the price is to low or too high the proba get's down, the pic price is arrount ln(price) = 5, the more we move from the optimal price the more a smoothing correction applied
             lead_time, the more the lead gets the reservation in advance, the more that the chance to converte is smaller.
             geo, more weight to canadian leads
             saisonality, cyclic variation during the year
        Output: pd.Dataframe()
    """
    rng = np.random.default_rng(42)
    n = 50_000
    start = pd.Timestamp("2026-01-01")
    ts = start + pd.to_timedelta(rng.uniform(0, 180, n), unit="D")
    partner_ids = [f"P{i:03d}" for i in range(60)]
    partner_eff = dict(zip(partner_ids, np.random.default_rng(2).normal(0, 0.5, 60)))
    partners = rng.choice(partner_ids, n, p=np.random.default_rng(1).dirichlet(np.ones(60)*0.6))
    device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.62, .31, .07])
    geo = rng.choice(["CA", "US", "FR", "UK", "DE", "other"], n, p=[.18, .34, .12, .11, .08, .17])
    # price generated with the lognormal to avoid negative values and insure a long tail for higher prices
    price = np.round(np.exp(rng.normal(5.1, 0.6, n)), 2)
    # lead_time to converte, it's a gamma tail 
    lead_time = np.maximum(0, rng.gamma(2.0, 12.0, n)).astype(int)
    nights = 1 + rng.poisson(2.0, n)
    past_clicks = rng.poisson(1.2, n)
    # logit : bias started at -3.4 if the other factors will be at 0, the base convertion propability will be samll, this is reflects the reality of otas
    #         for the 0.8 (device == 'desktop'), we simulate the observed bias in e commerce or online browsing in general, people are more willing to converte when using their laptop
    #         0.55 * np.log1p(past_clicks), np.log1p (compute(ln(1+x))), the more the lead clicks the more the chance to converte, but the effect of the first clic is huge comparing to 50 or 51 th one
    #         optimal price, -0.8 (log(price)- 5), non linear relation ship, when the price is to low or too high the proba get's down, the pic price is arrount ln(price) = 5, the more we move from the optimal price the more a smoothing correction applied
    #.        lead_time, the more the lead gets the reservation in advance, the more that the chance to converte is smaller.
    #         geo, more weight to canadian leads
    # saisonality, cyclic variation during the year
    logit = (-3.4 + 0.8*(device == "desktop") + 0.55*np.log1p(past_clicks)
            - 0.8*np.abs(np.log(price)-5.0) - 0.022*lead_time
            + 0.25*(geo=="CA") + np.array([partner_eff[p] for p in partners])
            + 0.15*np.sin(ts.dayofyear/58.0))
    p = 1/(1+np.exp(-logit)); y = rng.binomial(1, p)
    df = pd.DataFrame({"click_ts": ts, "partner_id": partners, "device": device, "geo": geo, 
                    "price": price, "lead_time_days": lead_time, "nights": nights, 
                    "past_click_30d": past_clicks, "converted": y})
    df["booking_amount"] = np.where(df.converted==1, df.price*df.nights*rng.uniform(0.9, 1.1, n), np.nan)
    df["minutes_on_ota_site"] = np.where(df.converted==1, rng.gamma(3, 4, n), rng.gamma(1.2, 2, n))
    df.loc[rng.choice(n, 1500, replace=False), "geo"] = None
    df = pd.concat([df, df.sample(300, random_state=0)], ignore_index=True)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    print(df.shape)
    df.head()
    return df 

# before the split make the proprocessing function that can help to see the data quality, cleaning
# add some feature enginnering to add some features for this speficic data.
#def make_splits(df): ...
#def train_lgbm(Xtr, ytr, Xva, yva, **params): ...
#def evaluate(model, X, y) -> dict: ...   # AUC, PR-AUC, Brier
#def run_experiment(params) -> dict: ...  # assemble et retourne une ligne de tableau
# make other models and run the experiments. 


# let's check the quality the univariate AUC in the data
from sklearn.metrics import roc_auc_score

def data_cont_leakage_detection(num_cols, df, target, target_rate, na_columns_serie): 
    target = df.converted

    bound_null = (target_rate) * .25

    list_col_leakage_null = []
    list_col_leakage_auc = []

    for col in df.columns:
        null_pct = na_columns_serie.get(col, 0.0)
        print(f"{col}, {null_pct}")
        if (1-target_rate) - bound_null <= null_pct/100.0 <= (1-target_rate) + bound_null:
            list_col_leakage_null.append(col)
        if col in num_cols:
            if col == "converted":
                continue
            col_data = df[col].rank(na_option='bottom')
            auc = roc_auc_score(target, col_data)
            print(f"{col:>22s} univariate AUC = {auc:.3f}")
            if 0 <= auc <= .1 or .9 <= auc <= 1:
                print(
                    f"{col} is suspected to be filled at convergence time, AUC :{auc}"
                )
                list_col_leakage_auc.append(col)
    # leakage potentiel 
    leakage_intersection = list(set(list_col_leakage_null) | set(list_col_leakage_auc))
    return leakage_intersection

# compute the information value to include all type of variables
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

def compute_iv_auc(df, cat_col, target_col):
    print("Processing categorical column:", cat_col)
    s_cat = df[cat_col].astype("category")
    df_count = df.groupby(s_cat, dropna=False)[target_col].agg(["count", "sum"])
    df_count["goods"] = df_count["sum"]
    df_count["bads"] = df_count["count"] - df_count["sum"]
    # get the propotion of goods and bads
    prop_goods = df_count["goods"] / df_count["goods"].sum()
    prop_bads = df_count["bads"] / df_count["bads"].sum()
    prop_goods = np.where(prop_goods == 0, .00001, prop_goods)
    prop_bads = np.where(prop_bads == 0, .0001, prop_bads)
    woe = np.log(prop_goods / prop_bads)
    iv = np.sum((prop_goods - prop_bads) * woe)
    # Calculate the AUC 
    model = HistGradientBoostingClassifier(
        max_iter=50, categorical_features=[True], random_state=42
    )
    y_probs = cross_val_predict(
        model, s_cat.to_frame(), df[target_col], cv=5, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(df[target_col], y_probs)
    return pd.Series({"IV": iv, "AUC": auc})

def make_split(df, split_col, low_pct, high_pct):
    c1, c2 = df[split_col].quantile(low_pct), df[split_col].quantile(high_pct)
    train = df[df[split_col] <= c1]
    valid = df[(df[split_col] > c1) & (df[split_col] <= c2)]
    test = df[df[split_col] > c2]
    for name, d in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name:5s} {len(d):6,d} | {d.click_ts.min().date()} -> {d.click_ts.max().date()} | rate {d.converted.mean():.4f}")
    return train, valid, test

# train function lgbm 
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def train_lgbm(Xtr, ytr, Xva, yva, num_cols, cat_cols, **params):
    Xtr_filtered = Xtr[cat_cols + num_cols].copy()
    Xva_filtered = Xva[cat_cols + num_cols].copy()

    default_params = {
        "n_estimators":400,
        "learning_rate":0.05,
        "verbose":-1,
        "num_leaves":31,
        "min_child_samples":50,
        "random_state":0,
    }

    params_f = {**default_params, **params}

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat_cols
            ),
            ("num", "passthrough", num_cols),
        ]
    )
    lgb_model = lgb.LGBMClassifier(**params_f)

    full_pipeline_lgbm = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lgb_model)
        ]
    )
    full_pipeline_lgbm.fit(Xtr_filtered, ytr)
    pa_val = full_pipeline_lgbm.predict_proba(Xva_filtered)[:,1]
    return full_pipeline_lgbm, pa_val

def train_lgr(Xtr, ytr, Xva, yva, num_cols, low_card_cols, high_card_cols, **params):

    default_param = {
        "C":1.0,
        "max_iter":1000,
        "random_state":0,
        "solver":"lbfgs",
    }
    params_f = {**default_param, **params}

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
            (
                "low_card",
                OneHotEncoder(handle_unknown="ignore"),
                low_card_cols,
            ),
            (
                "high_card",
                TargetEncoder(smooth="auto", cv=5),
                high_card_cols
            )
        ]
    )
    full_pipeline_lgr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(**params_f)
            )
        ]
            
    )
    full_pipeline_lgr.fit(Xtr, ytr)
    pa_val = full_pipeline_lgr.predict_proba(Xva)[:, 1]
    return full_pipeline_lgr, pa_val

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

def evaluate(model, X, y):

    p_te = model.predict_proba(X)[:, 1]
    real_rate = float(np.mean(y))
    p_mean = float(np.mean(p_te))
    brier = float(brier_score_loss(y, p_te))
    auc = float(roc_auc_score(y, p_te))
    loss = float(log_loss(y, p_te))

    return {
        "real_rate": round(real_rate, 4),
        "p_mean": round(p_mean, 4),
        "brier": round(brier, 4),
        "log_loss": round(loss, 4),
        "auc": round(auc, 4),
    }


def run_experiment(name, model_name, 
                   Xtr, 
                   ytr,
                   Xva, 
                   yva, 
                   Xte,
                   yte,
                   num_cols, 
                   cat_cols, 
                   low_card_cols,
                   high_card_cols,
                   # changing params
                   model_params=None):
    model_params = model_params or {}
    
    if model_name == "lgb":
        model, p_val = train_lgbm(Xtr, ytr, Xva, yva, num_cols, cat_cols, **model_params)
    elif model_name == "lgr":
        model, p_val = train_lgr(Xtr, ytr, Xva, yva, num_cols, low_card_cols, high_card_cols, **model_params)
        
    else:  
         raise ValueError(
                    f"Model {model_name} is not handled. Possible choices: 'lbg', 'lgr'"
                )
    metrics = evaluate(model, Xte, yte)

    return {"experiment name": name, 
            "model_name": model_name, 
            "metrics": metrics,
            **model_params}




