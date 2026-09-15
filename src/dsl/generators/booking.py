import numpy as np
import pandas as pd

def generate_booking_df(
    n_samples: int = 50_000, seed: int = 42
) -> tuple[pd.DataFrame, dict]:

    """Generates a synthetic e-commerce booking dataset. 
    Simulate clickstream data, user attributes, price sensitivities, and non-linear convertion probabilities
    using a logit model with speficic domain heuristics.
    Introduces missing values and ducplicate record to mimic real-work data quality

    Parameters
    ----------
    n_sample : int
            Number of primary sythetic sessions to generate.
    seed : int
            Random seed for full reproducibility of the data generation process.
        
        Returns
        -------
        pd.Dataframe 
            A Dataframe containing user sessions, booking details, conversion status, and simulated post-convertion metrics.

        Logit Conversion Model Mechanics
        --------------------------------
        - Base bias (-3.4): Reflects low overall baseline convertion rates.
        - Device Effects (+.8): Simulates higher conversion probability on desktop.
        - Past Clicks (+.55 * log1p): Diminishing marginal returns for user engagement.
        - Price Elasticity (-0.8 * log(price) - 5): Non linear penalty centred around optimal price.
        - Lead time (-0.022/day): decreasin convertion likelihood for booking made for bookings made far in advance.
        - Geo (+0.25): Positive conversion bias for Canadian leads.
        - Partner Effect: Random intercepts simulating heterogeneous partner performance.
        - Seasonality (+0.15 * sin): cyclic variation across the day of the year.
    """

    rng = np.random.default_rng(seed)

    #1. Temporal and partner attributes
    start_date = pd.Timestamp("2026-01-01")

    day_offset = rng.integers(0, 180, n_samples)

    # hours simulated with a day profil
    hour_weights = np.array([
        1, 1, 1, 1, 1, 2,      # 0-5h : night
        3, 5, 7, 8, 8, 8,      # 6-11h : morning
        9, 8, 7, 7, 8, 9,      # 12-17h : after noon
        11, 13, 14, 12, 8, 4,  # 18-23h : pic at the evening 
    ], dtype=float)

    hour = rng.choice(24, size=n_samples, p=hour_weights / hour_weights.sum())
    minute = rng.uniform(0, 60, n_samples)

    click_ts = (
        start_date
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )

    partner_ids = [f"P{i:03d}" for i in range(60)]
    partner_dirichlet_p = rng.dirichlet(np.ones(60) * 0.6)
    partner_eff_map = dict(zip(partner_ids, rng.normal(0, 0.5, 60)))

    partners = rng.choice(partner_ids, size=n_samples, p=partner_dirichlet_p)
    partner_effects = pd.Series(partners).map(partner_eff_map).to_numpy()

    #2. Categorical and numrical features
    device = rng.choice(
        ["mobile", "desktop", "tablet"], size=n_samples, p=[.62, .31, .07]
    )
    geo = rng.choice(
        ["CA", "US", "FR", "UK", "DE", "other"],
        size=n_samples,
        p=[.18, .34, .12, .11, .08, .17]
    )

    price = np.round(np.exp(rng.normal(5.1, 0.6, n_samples)), 2)
    lead_time_days = np.maximum(0, rng.gamma(2.0, 12.0, n_samples)).astype(int)
    nights = 1 + rng.poisson(2.0, n_samples)
    past_clicks_30d = rng.poisson(1.2, n_samples)

    #3. Non linear logit and target
    logit = (
        -3.4 
        + .8 * (device == "desktop")
        + .55 * np.log1p(past_clicks_30d)
        - .8 * np.abs(np.log(price) - 5.0)
        - 0.022 * lead_time_days
        + .25 * (geo == "CA")
        + partner_effects 
        + .15 * np.sin(click_ts.dayofyear / 58.0) # saisonality effect
        + .3 * np.sin(2 * np.pi * (hour - 15) / 24) # hour effect
    )

    conversion_prob = 1 / (1 + np.exp(-logit))
    converted = rng.binomial(1, conversion_prob)

    #4. Assembly
    df = pd.DataFrame(
        {"click_ts": click_ts, "partner_id": partners, "device": device, "geo": geo, 
        "price": price, "lead_time_days": lead_time_days, "nights": nights, "past_click_30d": past_clicks_30d,
        "converted": converted
        }
    )

    #5. Generate few data
    df["booking_amount"] = np.where(df.converted == 1, df.price*df.nights*rng.uniform(.9, 1.1, n_samples), np.nan)
    df["minutes_on_ots_site"] = np.where(df.converted == 1, rng.gamma(3, 4, n_samples), rng.gamma(1.2, 2, n_samples))
    df.loc[rng.choice(n_samples, 1500, replace=False), "geo"] = None
    df = pd.concat([df, df.sample(300, random_state=seed)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    schema =  {
        "num": ["price", "booking_amount", "minutes_on_ots_site", "nights", "past_click_30d", "lead_time_days"],
        "cat": ["device", "geo"],
        "ids": ["partner_id"],
        "target": "converted",
        "timestamp": "click_ts",
    }

    return df, schema
