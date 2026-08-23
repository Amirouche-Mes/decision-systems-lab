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
    partner_ids = [f"P{i:03}" for i in range(60)]
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
            - 0.8*np.abs(np.log(price)-0.5) - 0.0228*lead_time
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


