"""
California ZIP-code precipitation and housing analysis (2015-2025)

This script downloads Zillow's public ZIP-level ZHVI CSV and Census ZIP Code
Tabulation Area centroids, retrieves monthly precipitation from Open-Meteo for
each available California ZCTA centroid, builds a ZIP-month panel, and estimates
fixed-effects regressions. It can optionally merge a user-collected monthly
sentiment CSV.

Important:
- ZCTAs are Census statistical approximations to ZIP codes, not exact USPS ZIP boundaries.
- Open-Meteo ERA5/ERA5-Land is gridded reanalysis, not a direct rain-gauge observation.
- The sentiment module is optional; do not label news sentiment as social-media sentiment.
- This script must be run in an environment with internet access.
"""

import io, time, zipfile, warnings
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = Path("ca_precip_housing_output")
OUT.mkdir(exist_ok=True)
START = "2015-01-01"
END = "2025-12-31"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CaliforniaHousingWeatherResearch/1.0 (academic analysis)"})

ZILLOW_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip/"
    "zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
)
# Census ZCTA Gazetteer centroids, 2020 vintage
CENSUS_ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip"

def download(url, timeout=90):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def load_zillow():
    print("Downloading Zillow ZIP-level ZHVI...")
    raw = pd.read_csv(io.BytesIO(download(ZILLOW_URL)), low_memory=False)
    # Zillow files usually have RegionName, StateName and monthly columns.
    if "StateName" not in raw.columns or "RegionName" not in raw.columns:
        raise ValueError(f"Unexpected Zillow columns: {raw.columns[:20].tolist()}")
    raw = raw[raw["StateName"].eq("CA")].copy()
    date_cols = [c for c in raw.columns if str(c)[:4].isdigit() and "-" in str(c)]
    dates = pd.to_datetime(date_cols, errors="coerce")
    keep = [(c, d) for c, d in zip(date_cols, dates) if pd.notna(d) and START <= d.strftime("%Y-%m-%d") <= END]
    long = raw.melt(
        id_vars=["RegionName", "StateName"],
        value_vars=[c for c, _ in keep],
        var_name="date", value_name="zhvi"
    )
    long["date"] = pd.to_datetime(long["date"])
    long["zip"] = long["RegionName"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    long["zhvi"] = pd.to_numeric(long["zhvi"], errors="coerce")
    long = long[["zip", "date", "zhvi"]].dropna(subset=["zhvi"])
    long["home_value_growth_pct"] = long.groupby("zip")["zhvi"].pct_change() * 100
    long.to_csv(OUT / "california_zhvi_zip_month.csv", index=False)
    print(f"ZHVI rows: {len(long):,}; ZIPs: {long.zip.nunique():,}")
    return long

def load_zcta_centroids():
    print("Downloading Census ZCTA centroids...")
    zf = zipfile.ZipFile(io.BytesIO(download(CENSUS_ZCTA_URL)))
    name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
    gaz = pd.read_csv(zf.open(name), sep="|", dtype={"GEOID": str})
    # Gazetteer uses INTPTLAT and INTPTLONG columns.
    gaz = gaz.rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    gaz["zip"] = gaz["zip"].astype(str).str.zfill(5)
    gaz["lat"] = pd.to_numeric(gaz["lat"], errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz["lon"], errors="coerce")
    # California ZCTAs are identified by overlap with CA ZHVI ZIPs below.
    return gaz[["zip", "lat", "lon"]].dropna()

def fetch_monthly_weather(zip_df):
    print("Retrieving monthly precipitation from Open-Meteo (this can take time)...")
    rows = []
    # One location per request is robust to API response formats/rate limits.
    for i, r in enumerate(zip_df.itertuples(index=False), start=1):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": float(r.lat), "longitude": float(r.lon),
            "start_date": START, "end_date": END,
            "daily": "precipitation_sum", "timezone": "America/Los_Angeles",
            "models": "era5_land"
        }
        try:
            resp = SESSION.get(url, params=params, timeout=90)
            resp.raise_for_status()
            payload = resp.json()
            daily = pd.DataFrame(payload.get("daily", {}))
            if daily.empty or "time" not in daily or "precipitation_sum" not in daily:
                continue
            daily["date"] = pd.to_datetime(daily["time"])
            daily["zip"] = r.zip
            daily["precipitation_sum"] = pd.to_numeric(daily["precipitation_sum"], errors="coerce")
            daily["extreme_rain_day"] = daily["precipitation_sum"] >= 25.0
            daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
            monthly = daily.groupby(["zip", "month"], as_index=False).agg(
                precipitation_mm=("precipitation_sum", "sum"),
                extreme_rain_days=("extreme_rain_day", "sum"),
                precipitation_days=("precipitation_sum", lambda s: int((s >= 1.0).sum()))
            )
            rows.append(monthly)
        except Exception as e:
            warnings.warn(f"Weather failed for ZIP/ZCTA {r.zip}: {e}")
        if i % 25 == 0:
            print(f"  weather locations processed: {i}/{len(zip_df)}")
        # Be polite to public API
        time.sleep(0.15)
    if not rows:
        raise RuntimeError("No weather observations retrieved. Check network/API access.")
    weather = pd.concat(rows, ignore_index=True)
    weather.to_csv(OUT / "california_precipitation_zip_month.csv", index=False)
    print(f"Weather rows: {len(weather):,}; ZIPs: {weather.zip.nunique():,}")
    return weather

def main(sentiment_csv=None):
    housing = load_zillow()
    centroids = load_zcta_centroids()
    valid_zips = set(housing["zip"].unique())
    geo = centroids[centroids["zip"].isin(valid_zips)].copy()
    print(f"ZCTA centroids matched to Zillow ZIPs: {len(geo):,}")
    weather = fetch_monthly_weather(geo)
    panel = housing.merge(weather, left_on=["zip", "date"], right_on=["zip", "month"], how="inner")
    panel = panel.drop(columns=["month"])
    # Rainfall anomaly relative to each ZIP's calendar-month baseline.
    panel["calendar_month"] = panel["date"].dt.month
    clim = panel.groupby(["zip", "calendar_month"])["precipitation_mm"].transform("mean")
    panel["rainfall_anomaly_mm"] = panel["precipitation_mm"] - clim
    panel["rainfall_anomaly_pct"] = np.where(clim > 0, 100 * panel["rainfall_anomaly_mm"] / clim, np.nan)
    panel["year_month"] = panel["date"].dt.strftime("%Y-%m")

    # Optional sentiment CSV schema:
    # zip, date (month timestamp), negative_sentiment (0-1), flood_topic_share (0-1)
    if sentiment_csv:
        sent = pd.read_csv(sentiment_csv, dtype={"zip": str})
        sent["zip"] = sent["zip"].str.zfill(5)
        sent["date"] = pd.to_datetime(sent["date"]).dt.to_period("M").dt.to_timestamp()
        panel = panel.merge(sent, on=["zip", "date"], how="left")

    panel.to_csv(OUT / "ca_zip_month_panel.csv", index=False)
    print(f"Panel rows: {len(panel):,}; ZIPs: {panel.zip.nunique():,}")

    # Main analysis: ZIP fixed effects and calendar-month fixed effects.
    # Cluster standard errors by ZIP. Growth outcome is percentage points.
    base = panel.dropna(subset=["home_value_growth_pct", "rainfall_anomaly_mm"]).copy()
    if base["zip"].nunique() >= 2 and len(base) >= 100:
        m1 = smf.ols(
            "home_value_growth_pct ~ rainfall_anomaly_mm + extreme_rain_days + C(zip) + C(year_month)",
            data=base
        ).fit(cov_type="cluster", cov_kwds={"groups": base["zip"]})
        with open(OUT / "model_1_weather_only.txt", "w") as f:
            f.write(m1.summary().as_text())
        print("\nMODEL 1: Weather-only fixed effects")
        print(m1.summary().tables[1])

        # Lagged rainfall test
        base = base.sort_values(["zip", "date"])
        base["rain_anomaly_lag1"] = base.groupby("zip")["rainfall_anomaly_mm"].shift(1)
        lag = base.dropna(subset=["rain_anomaly_lag1"])
        mlag = smf.ols(
            "home_value_growth_pct ~ rainfall_anomaly_mm + rain_anomaly_lag1 + extreme_rain_days + C(zip) + C(year_month)",
            data=lag
        ).fit(cov_type="cluster", cov_kwds={"groups": lag["zip"]})
        with open(OUT / "model_2_lagged_weather.txt", "w") as f:
            f.write(mlag.summary().as_text())
        print("\nMODEL 2: Current + one-month lag")
        print(mlag.summary().tables[1])

        if {"negative_sentiment", "flood_topic_share"}.issubset(panel.columns):
            s = panel.dropna(subset=["home_value_growth_pct", "rainfall_anomaly_mm", "negative_sentiment"])
            if len(s) >= 100:
                m3 = smf.ols(
                    "home_value_growth_pct ~ rainfall_anomaly_mm * negative_sentiment + flood_topic_share + extreme_rain_days + C(zip) + C(year_month)",
                    data=s
                ).fit(cov_type="cluster", cov_kwds={"groups": s["zip"]})
                with open(OUT / "model_3_weather_plus_sentiment.txt", "w") as f:
                    f.write(m3.summary().as_text())
                print("\nMODEL 3: Weather + sentiment interaction")
                print(m3.summary().tables[1])
            else:
                print("Sentiment model skipped: insufficient merged observations.")
        else:
            print("Sentiment model not run: supply an authorized, geographically/time-aligned sentiment CSV.")
    else:
        print("Not enough merged observations to estimate fixed-effects models.")
    print(f"\nOutputs saved to: {OUT.resolve()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment-csv", default=None, help="Optional CSV with zip,date,negative_sentiment,flood_topic_share")
    args = parser.parse_args()
    main(args.sentiment_csv)
