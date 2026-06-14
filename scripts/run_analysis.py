"""
run_analysis.py
---------------
Local-only analysis script for:

    伊以冲突对国际油价的冲击效应研究

This script reads only four local CSV files:

    data/raw/events_manual.csv
    data/raw/fred_wti.csv
    data/raw/fred_brent.csv
    data/raw/fred_sp500.csv

Main tasks:
    1. Load and clean local FRED WTI, Brent and S&P 500 data.
    2. Load manually selected Iran-Israel conflict events.
    3. Build data/processed/oil_market_dataset.csv.
    4. Run event study:
         market model -> abnormal return -> CAR -> CAR t-test.
    5. Run interrupted time series analysis:
         single-event ITS and multiple-event ITS.
    6. Run residual diagnostics:
         DW / BP / BG tests and Newey-West HAC correction.
    7. Run multiple structural breakpoint detection using ruptures.
    8. Generate figures used in report.md.

Required packages:
    numpy
    pandas
    matplotlib
    scipy
    statsmodels
    ruptures
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey

import ruptures as rpt


warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths and parameters
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"

FRED_WTI_PATH = DATA_RAW / "fred_wti.csv"
FRED_BRENT_PATH = DATA_RAW / "fred_brent.csv"
FRED_SP500_PATH = DATA_RAW / "fred_sp500.csv"
EVENTS_PATH = DATA_RAW / "events_manual.csv"

ESTIMATION_WINDOW = 120
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10
ITS_WINDOW = 60
SIGNIFICANCE_LEVEL = 0.05
N_BREAKPOINTS = 7

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. General utilities
# ============================================================

def ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved: {path.relative_to(PROJECT_ROOT)}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n  {path}\n"
            f"请确认该文件已经放在 data/raw/ 文件夹中。"
        )


def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "Date",
        "date",
        "DATE",
        "observation_date",
        "Observation Date",
        "observation date",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find date column. Existing columns: {df.columns.tolist()}")


def significance_mark(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def nearest_trading_day(index: pd.DatetimeIndex, event_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    If event_date is not a trading day in the dataset, use the next available trading day.
    """
    index = pd.DatetimeIndex(index).sort_values()
    event_date = pd.Timestamp(event_date)

    if event_date in index:
        return event_date

    future_dates = index[index >= event_date]
    if len(future_dates) == 0:
        return None

    return future_dates[0]


def safe_add_constant(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


def drop_exact_collinear_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop exactly collinear columns using incremental matrix-rank check.

    This does not remove highly correlated variables, only variables that cause exact
    rank deficiency. It makes the multiple ITS model more stable while keeping the
    design as close as possible to the intended specification.
    """
    kept_cols: list[str] = []
    dropped_cols: list[str] = []

    current = pd.DataFrame(index=X.index)

    current_rank = 0
    for col in X.columns:
        test = pd.concat([current, X[[col]]], axis=1)
        new_rank = np.linalg.matrix_rank(test.to_numpy(dtype=float))

        if new_rank > current_rank:
            kept_cols.append(col)
            current = test
            current_rank = new_rank
        else:
            dropped_cols.append(col)

    return X[kept_cols], dropped_cols


# ============================================================
# 3. Data loading
# ============================================================

def normalize_fred_csv(path: Path, value_name: str, possible_series_ids: list[str]) -> pd.DataFrame:
    """
    Read manually downloaded FRED CSV and normalize it to:

        Date, value_name

    Supported common FRED formats include:
        observation_date,DCOILWTICO
        DATE,DCOILWTICO
        Date,DCOILWTICO

    Missing values such as "." are converted to NaN and dropped.
    """
    require_file(path)

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = find_date_column(df)

    value_col = None
    for c in possible_series_ids + [value_name]:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        non_date_cols = [c for c in df.columns if c != date_col]
        if len(non_date_cols) == 1:
            value_col = non_date_cols[0]
        else:
            raise ValueError(
                f"Cannot determine value column in {path.name}. "
                f"Existing columns: {df.columns.tolist()}"
            )

    out = df[[date_col, value_col]].copy()
    out.columns = ["Date", value_name]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")

    out = (
        out.dropna(subset=["Date", value_name])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return out


def load_events(path: Path) -> pd.DataFrame:
    """
    Read events_manual.csv.

    Preferred columns:
        event_id, date, event_name, description, type

    The function also accepts simple alternatives:
        id -> event_id
        name -> event_name
        日期 -> date
        事件 -> event_name
        类型 -> type
        描述 -> description
    """
    require_file(path)

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        c_lower = c.lower()

        if c_lower in ["id", "eventid", "event_id", "编号"]:
            rename_map[c] = "event_id"
        elif c_lower in ["date", "event_date", "日期", "事件日期"]:
            rename_map[c] = "date"
        elif c_lower in ["event_name", "name", "event", "事件", "事件名称"]:
            rename_map[c] = "event_name"
        elif c_lower in ["description", "desc", "事件描述", "描述"]:
            rename_map[c] = "description"
        elif c_lower in ["type", "事件类型", "类型"]:
            rename_map[c] = "type"

    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        date_col = find_date_column(df)
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "event_id" not in df.columns:
        df["event_id"] = [f"E{i + 1}" for i in range(len(df))]

    if "event_name" not in df.columns:
        df["event_name"] = df["event_id"]

    if "description" not in df.columns:
        df["description"] = df["event_name"]

    if "type" not in df.columns:
        df["type"] = "event"

    df["event_id"] = df["event_id"].astype(str)
    df["event_name"] = df["event_name"].astype(str)
    df["description"] = df["description"].astype(str)
    df["type"] = df["type"].astype(str)

    df = df[["event_id", "date", "event_name", "description", "type"]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print("Loading local CSV files")
    print("=" * 72)

    wti = normalize_fred_csv(FRED_WTI_PATH, "WTI", ["DCOILWTICO"])
    brent = normalize_fred_csv(FRED_BRENT_PATH, "Brent", ["DCOILBRENTEU"])
    sp500 = normalize_fred_csv(FRED_SP500_PATH, "SP500", ["SP500"])
    events = load_events(EVENTS_PATH)

    df = wti.merge(brent, on="Date", how="inner")
    df = df.merge(sp500, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Merged dataset is empty. 请检查三个 FRED CSV 的日期区间是否重叠。")

    df["WTI_return"] = np.log(df["WTI"] / df["WTI"].shift(1))
    df["Brent_return"] = np.log(df["Brent"] / df["Brent"].shift(1))
    df["Market_return"] = np.log(df["SP500"] / df["SP500"].shift(1))
    df["Brent_WTI_spread"] = df["Brent"] - df["WTI"]

    df = df.dropna(subset=["WTI_return", "Brent_return", "Market_return"]).reset_index(drop=True)

    save_csv(df, DATA_PROCESSED / "oil_market_dataset.csv")

    print(f"WTI observations:   {len(wti)}")
    print(f"Brent observations: {len(brent)}")
    print(f"SP500 observations: {len(sp500)}")
    print(f"Merged observations:{len(df)}")
    print(f"Data range:         {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Events:             {len(events)}")
    print()

    return df, events


# ============================================================
# 4. Event study
# ============================================================

class EventStudy:
    """
    Event study using market model:

        R_oil,t = alpha + beta * R_market,t + epsilon_t

    Abnormal return:
        AR_t = R_oil,t - predicted normal return

    Cumulative abnormal return:
        CAR = sum(AR_t) over event window
    """

    def __init__(
        self,
        data: pd.DataFrame,
        events: pd.DataFrame,
        return_col: str = "WTI_return",
        market_col: str = "Market_return",
        estimation_window: int = ESTIMATION_WINDOW,
        pre_event: int = EVENT_WINDOW_PRE,
        post_event: int = EVENT_WINDOW_POST,
    ):
        self.data = data.copy()
        self.events = events.copy()
        self.return_col = return_col
        self.market_col = market_col
        self.estimation_window = estimation_window
        self.pre_event = pre_event
        self.post_event = post_event

        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data = self.data.sort_values("Date").set_index("Date")

        self.results: dict[str, dict] = {}

    def analyze_one_event(self, event_row: pd.Series) -> Optional[dict]:
        event_id = str(event_row["event_id"])
        event_name = str(event_row["event_name"])
        event_date_original = pd.Timestamp(event_row["date"])

        event_date_trading = nearest_trading_day(self.data.index, event_date_original)
        if event_date_trading is None:
            print(f"skip {event_id}: no trading day after event date")
            return None

        event_idx = self.data.index.get_loc(event_date_trading)

        est_start = max(0, event_idx - self.pre_event - self.estimation_window)
        est_end = event_idx - self.pre_event

        if est_end - est_start < 30:
            print(f"skip {event_id}: insufficient estimation window")
            return None

        window_start = max(0, event_idx - self.pre_event)
        window_end = min(len(self.data), event_idx + self.post_event + 1)

        est_data = self.data.iloc[est_start:est_end].dropna(subset=[self.return_col, self.market_col])
        event_data = self.data.iloc[window_start:window_end].dropna(subset=[self.return_col, self.market_col])

        if len(est_data) < 30 or len(event_data) < 3:
            print(f"skip {event_id}: insufficient valid data")
            return None

        y_est = est_data[self.return_col]
        X_est = safe_add_constant(est_data[[self.market_col]])

        market_model = OLS(y_est, X_est).fit()

        X_event = safe_add_constant(event_data[[self.market_col]])
        normal_return = market_model.predict(X_event)

        ar = event_data[self.return_col] - normal_return
        car = ar.cumsum()

        residual_sigma = market_model.resid.std(ddof=1)
        n_window = len(ar)
        car_total = car.iloc[-1]

        if residual_sigma > 0 and n_window > 1:
            t_stat = car_total / (residual_sigma * np.sqrt(n_window))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_window - 1))
        else:
            t_stat = np.nan
            p_value = np.nan

        if event_date_trading in event_data.index:
            event_position = event_data.index.get_loc(event_date_trading)
        else:
            event_position = 0

        ar_event_day = ar.iloc[event_position]

        result = {
            "event_id": event_id,
            "event_name": event_name,
            "event_date_original": event_date_original,
            "event_date_trading": event_date_trading,
            "event_type": str(event_row["type"]),
            "AR": ar,
            "CAR": car,
            "AR_event_day": ar_event_day,
            "CAR_total": car_total,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significance": significance_mark(p_value),
            "significant_5pct": bool(p_value < SIGNIFICANCE_LEVEL) if pd.notna(p_value) else False,
            "alpha": market_model.params.get("const", np.nan),
            "beta": market_model.params.get(self.market_col, np.nan),
            "market_model_r2": market_model.rsquared,
            "sigma_resid": residual_sigma,
            "n_event_window": n_window,
            "event_window_start": event_data.index[0],
            "event_window_end": event_data.index[-1],
        }

        self.results[event_id] = result
        return result

    def run(self) -> pd.DataFrame:
        print("=" * 72)
        print("Event study: market model -> AR/CAR -> CAR t-test")
        print("=" * 72)
        print(f"Estimation window: {self.estimation_window} trading days")
        print(f"Event window:      [-{self.pre_event}, +{self.post_event}]")
        print("-" * 72)

        rows = []

        for _, event in self.events.iterrows():
            r = self.analyze_one_event(event)
            if r is None:
                continue

            print(
                f"{r['event_id']} {r['event_name']}: "
                f"CAR={r['CAR_total'] * 100:.3f}%, "
                f"t={r['t_statistic']:.3f}, "
                f"p={r['p_value']:.4f} {r['significance']}"
            )

            rows.append({
                "event_id": r["event_id"],
                "event_name": r["event_name"],
                "event_date_original": r["event_date_original"].strftime("%Y-%m-%d"),
                "event_date_trading": r["event_date_trading"].strftime("%Y-%m-%d"),
                "event_type": r["event_type"],
                "event_window_start": r["event_window_start"].strftime("%Y-%m-%d"),
                "event_window_end": r["event_window_end"].strftime("%Y-%m-%d"),
                "AR_event_day_pct": r["AR_event_day"] * 100,
                "CAR_pct": r["CAR_total"] * 100,
                "t_statistic": r["t_statistic"],
                "p_value": r["p_value"],
                "significance": r["significance"],
                "significant_5pct": r["significant_5pct"],
                "alpha": r["alpha"],
                "beta": r["beta"],
                "market_model_r2": r["market_model_r2"],
                "sigma_resid": r["sigma_resid"],
                "n_event_window": r["n_event_window"],
            })

        summary = pd.DataFrame(rows)
        save_csv(summary, OUTPUT_DIR / "event_study_results.csv")
        print()

        return summary


# ============================================================
# 5. Interrupted time series
# ============================================================

def fit_single_its(
    data: pd.DataFrame,
    events: pd.DataFrame,
    price_col: str = "WTI",
    window: int = ITS_WINDOW,
) -> pd.DataFrame:
    """
    Single-event ITS:

        Y_t = beta0 + beta1*T + beta2*D + beta3*T_after + epsilon_t

    For inference, use Newey-West HAC standard errors.
    """
    print("=" * 72)
    print("Single-event interrupted time series analysis")
    print("=" * 72)

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    rows = []

    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        event_name = str(event["event_name"])
        event_date_original = pd.Timestamp(event["date"])

        event_date_trading = nearest_trading_day(df.index, event_date_original)
        if event_date_trading is None:
            print(f"skip {event_id}: no trading day after event date")
            continue

        event_idx = df.index.get_loc(event_date_trading)
        start_idx = max(0, event_idx - window)
        end_idx = min(len(df), event_idx + window + 1)

        sub = df.iloc[start_idx:end_idx].copy()
        if len(sub) < 30:
            print(f"skip {event_id}: insufficient ITS window")
            continue

        sub["T"] = np.arange(len(sub))
        sub["D"] = (sub.index >= event_date_trading).astype(int)
        t0 = sub.loc[sub.index >= event_date_trading, "T"].iloc[0]
        sub["T_after"] = sub["D"] * (sub["T"] - t0)

        y = sub[price_col]
        X = safe_add_constant(sub[["T", "D", "T_after"]])

        model_ols = OLS(y, X).fit()
        model_hac = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        resid = model_ols.resid
        dw = durbin_watson(resid)

        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, X)
        except Exception:
            bp_stat, bp_pvalue = np.nan, np.nan

        try:
            bg_stat, bg_pvalue, _, _ = acorr_breusch_godfrey(model_ols, nlags=min(5, len(sub) // 5))
        except Exception:
            bg_stat, bg_pvalue = np.nan, np.nan

        level_change = model_hac.params.get("D", np.nan)
        level_p = model_hac.pvalues.get("D", np.nan)
        slope_change = model_hac.params.get("T_after", np.nan)
        slope_p = model_hac.pvalues.get("T_after", np.nan)

        rows.append({
            "event_id": event_id,
            "event_name": event_name,
            "event_date_original": event_date_original.strftime("%Y-%m-%d"),
            "event_date_trading": event_date_trading.strftime("%Y-%m-%d"),
            "window_start": sub.index[0].strftime("%Y-%m-%d"),
            "window_end": sub.index[-1].strftime("%Y-%m-%d"),
            "level_change": level_change,
            "level_change_pvalue": level_p,
            "level_change_sig": significance_mark(level_p),
            "slope_change": slope_change,
            "slope_change_pvalue": slope_p,
            "slope_change_sig": significance_mark(slope_p),
            "durbin_watson": dw,
            "bp_statistic": bp_stat,
            "bp_pvalue": bp_pvalue,
            "bg_statistic": bg_stat,
            "bg_pvalue": bg_pvalue,
            "r_squared": model_hac.rsquared,
            "adj_r_squared": model_hac.rsquared_adj,
            "n_obs": len(sub),
        })

        print(
            f"{event_id} {event_name}: "
            f"level={level_change:.3f} p={level_p:.4f} {significance_mark(level_p)}, "
            f"slope={slope_change:.4f} p={slope_p:.4f} {significance_mark(slope_p)}, "
            f"DW={dw:.3f}"
        )

    out = pd.DataFrame(rows)
    save_csv(out, OUTPUT_DIR / "its_single_results.csv")
    print()

    return out


def fit_multiple_its(
    data: pd.DataFrame,
    events: pd.DataFrame,
    price_col: str = "WTI",
) -> tuple[pd.DataFrame, pd.DataFrame, object, object, pd.Series, pd.DataFrame, list[str]]:
    """
    Multiple-event ITS.

    Intended full specification:

        Y_t = beta0 + beta1*T
              + sum_i gamma_i * D_i
              + sum_i delta_i * T_after_i
              + epsilon_t

    Because event dummy variables and post-event trends can be exactly collinear
    in some event configurations, the function removes exact collinear columns
    before fitting the model.
    """
    print("=" * 72)
    print("Multiple-event interrupted time series analysis")
    print("=" * 72)

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    df["T"] = np.arange(len(df))

    event_map = []

    for _, event in events.sort_values("date").iterrows():
        event_id = str(event["event_id"])
        event_name = str(event["event_name"])
        event_date_original = pd.Timestamp(event["date"])

        event_date_trading = nearest_trading_day(df.index, event_date_original)
        if event_date_trading is None:
            continue

        d_col = f"D_{event_id}"
        t_col = f"T_after_{event_id}"

        df[d_col] = (df.index >= event_date_trading).astype(int)

        t0 = df.loc[df.index >= event_date_trading, "T"].iloc[0]
        df[t_col] = df[d_col] * (df["T"] - t0)

        event_map.append({
            "event_id": event_id,
            "event_name": event_name,
            "event_date_original": event_date_original.strftime("%Y-%m-%d"),
            "event_date_trading": event_date_trading.strftime("%Y-%m-%d"),
            "D_col": d_col,
            "T_after_col": t_col,
        })

    exog_cols = ["T"]
    for item in event_map:
        exog_cols.extend([item["D_col"], item["T_after_col"]])

    y = df[price_col]
    X_raw = safe_add_constant(df[exog_cols])
    X, dropped_cols = drop_exact_collinear_columns(X_raw)

    if dropped_cols:
        print("Dropped exact collinear variables:")
        for c in dropped_cols:
            print(f"  - {c}")

    model_ols = OLS(y, X).fit()
    model_hac = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    dw = durbin_watson(model_ols.resid)

    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(model_ols.resid, X)
    except Exception:
        bp_stat, bp_pvalue = np.nan, np.nan

    try:
        bg_stat, bg_pvalue, _, _ = acorr_breusch_godfrey(model_ols, nlags=10)
    except Exception:
        bg_stat, bg_pvalue = np.nan, np.nan

    rows = []
    for var in model_hac.params.index:
        rows.append({
            "variable": var,
            "coefficient": model_hac.params[var],
            "hac_std_error": model_hac.bse[var],
            "t_value": model_hac.tvalues[var],
            "p_value": model_hac.pvalues[var],
            "significance": significance_mark(model_hac.pvalues[var]),
            "r_squared": model_hac.rsquared,
            "adj_r_squared": model_hac.rsquared_adj,
            "durbin_watson": dw,
            "bp_statistic": bp_stat,
            "bp_pvalue": bp_pvalue,
            "bg_statistic": bg_stat,
            "bg_pvalue": bg_pvalue,
            "n_obs": int(model_hac.nobs),
            "dropped_exact_collinear_variables": "; ".join(dropped_cols),
        })

    out = pd.DataFrame(rows)
    save_csv(out, OUTPUT_DIR / "its_multiple_results.csv")

    model_data = df.reset_index()
    save_csv(pd.DataFrame(event_map), OUTPUT_DIR / "its_multiple_event_variables.csv")

    print(f"R²={model_hac.rsquared:.4f}, Adj.R²={model_hac.rsquared_adj:.4f}, DW={dw:.4f}")
    print(f"BP p-value={bp_pvalue:.4f}" if pd.notna(bp_pvalue) else "BP p-value=NA")
    print(f"BG p-value={bg_pvalue:.4f}" if pd.notna(bg_pvalue) else "BG p-value=NA")
    print()

    return out, model_data, model_hac, model_ols, y, X, dropped_cols


# ============================================================
# 6. Structural break detection
# ============================================================

def detect_breakpoints(
    data: pd.DataFrame,
    events: pd.DataFrame,
    price_col: str = "WTI",
    n_breakpoints: int = N_BREAKPOINTS,
) -> pd.DataFrame:
    """
    Multiple structural breakpoint detection using ruptures.

    Note:
        ruptures returns segment endpoints. Therefore the breakpoint date is
        mapped using bp - 1 to avoid shifting the breakpoint one observation ahead.
    """
    print("=" * 72)
    print("Multiple structural breakpoint detection")
    print("=" * 72)

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 80:
        print("Too few observations for breakpoint detection.")
        out = pd.DataFrame()
        save_csv(out, OUTPUT_DIR / "breakpoints.csv")
        return out

    signal = df[price_col].to_numpy(dtype=float).reshape(-1, 1)

    max_allowed = max(1, len(df) // 40)
    n_bkps = min(n_breakpoints, max_allowed)

    algo = rpt.Dynp(model="l2", min_size=20, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps)

    event_dates = [
        (str(row["event_id"]), str(row["event_name"]), pd.Timestamp(row["date"]))
        for _, row in events.iterrows()
    ]

    rows = []
    for i, bp in enumerate(bkps[:-1], start=1):
        idx = min(max(bp - 1, 0), len(df) - 1)
        bp_date = pd.Timestamp(df.loc[idx, "Date"])
        bp_price = df.loc[idx, price_col]

        nearest_event_id = ""
        nearest_event_name = ""
        nearest_event_date = pd.NaT
        diff_days = np.nan

        if event_dates:
            nearest = min(event_dates, key=lambda x: abs((bp_date - x[2]).days))
            nearest_event_id, nearest_event_name, nearest_event_date = nearest
            diff_days = (bp_date - nearest_event_date).days

        rows.append({
            "breakpoint_id": f"BP{i}",
            "breakpoint_date": bp_date.strftime("%Y-%m-%d"),
            f"{price_col}_price": bp_price,
            "nearest_event_id": nearest_event_id,
            "nearest_event_name": nearest_event_name,
            "nearest_event_date": nearest_event_date.strftime("%Y-%m-%d") if pd.notna(nearest_event_date) else "",
            "diff_days_breakpoint_minus_event": diff_days,
        })

        print(
            f"BP{i}: {bp_date.strftime('%Y-%m-%d')}, "
            f"{price_col}={bp_price:.2f}, "
            f"nearest={nearest_event_id}, diff={diff_days:+.0f} days"
        )

    out = pd.DataFrame(rows)
    save_csv(out, OUTPUT_DIR / "breakpoints.csv")
    print()

    return out


# ============================================================
# 7. Figures
# ============================================================

def plot_oil_prices_with_events(data: pd.DataFrame, events: pd.DataFrame) -> None:
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df["Date"], df["WTI"], label="WTI", linewidth=1.4)
    ax.plot(df["Date"], df["Brent"], label="Brent", linewidth=1.4)

    for _, e in events.iterrows():
        date = pd.Timestamp(e["date"])
        event_type = str(e["type"]).lower()
        linestyle = ":" if "de" in event_type or "缓和" in event_type else "--"
        ax.axvline(date, linestyle=linestyle, linewidth=1.0, alpha=0.75)
        ax.text(date, ax.get_ylim()[1] * 0.98, str(e["event_id"]), rotation=45, fontsize=8, va="top")

    ax.set_title("WTI/Brent 原油价格走势与伊以冲突关键事件")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_oil_price_events.png", dpi=150)
    plt.close()


def plot_car_comparison(es: EventStudy) -> None:
    if not es.results:
        return

    n = len(es.results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (event_id, r) in zip(axes, es.results.items()):
        car = r["CAR"]
        days = np.arange(-EVENT_WINDOW_PRE, -EVENT_WINDOW_PRE + len(car))

        ax.plot(days, car.values * 100, linewidth=2)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linestyle="--", linewidth=1.0)
        ax.fill_between(days, car.values * 100, 0, alpha=0.2)

        ax.set_title(
            f"{event_id}: {r['event_name']}\n"
            f"CAR={r['CAR_total'] * 100:.2f}%, p={r['p_value']:.3f}{r['significance']}",
            fontsize=9,
        )
        ax.set_xlabel("相对事件日")
        ax.set_ylabel("CAR (%)")
        ax.grid(alpha=0.3)

    for ax in axes[len(es.results):]:
        ax.set_visible(False)

    plt.suptitle("各事件累计异常收益率 CAR 走势对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_car_comparison.png", dpi=150)
    plt.close()


def plot_its_analysis(data: pd.DataFrame, events: pd.DataFrame) -> None:
    """
    Plot all single-event ITS fits.
    """
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    selected_events = events.copy()
    n = len(selected_events)

    if n == 0:
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, event) in zip(axes, selected_events.iterrows()):
        event_id = str(event["event_id"])
        event_name = str(event["event_name"])
        event_date_original = pd.Timestamp(event["date"])

        event_date_trading = nearest_trading_day(df.index, event_date_original)
        if event_date_trading is None:
            ax.set_visible(False)
            continue

        event_idx = df.index.get_loc(event_date_trading)
        start_idx = max(0, event_idx - ITS_WINDOW)
        end_idx = min(len(df), event_idx + ITS_WINDOW + 1)

        sub = df.iloc[start_idx:end_idx].copy()
        sub["T"] = np.arange(len(sub))
        sub["D"] = (sub.index >= event_date_trading).astype(int)
        t0 = sub.loc[sub.index >= event_date_trading, "T"].iloc[0]
        sub["T_after"] = sub["D"] * (sub["T"] - t0)

        y = sub["WTI"]
        X = safe_add_constant(sub[["T", "D", "T_after"]])
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        fitted = model.predict(X)

        level = model.params.get("D", np.nan)
        p = model.pvalues.get("D", np.nan)

        ax.scatter(sub.index, sub["WTI"], s=12, alpha=0.55, label="WTI")
        ax.plot(sub.index, fitted, linewidth=2, label="ITS 拟合")
        ax.axvline(event_date_trading, linestyle="--", linewidth=1.0)

        ax.set_title(f"{event_id}: 水平变化={level:.2f}{significance_mark(p)}", fontsize=10)
        ax.set_xlabel("日期")
        ax.set_ylabel("WTI")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("中断时间序列 ITS：单事件窗口拟合", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_its_analysis.png", dpi=150)
    plt.close()


def plot_breakpoints(data: pd.DataFrame, events: pd.DataFrame, breakpoints: pd.DataFrame) -> None:
    if breakpoints.empty:
        return

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df["Date"], df["WTI"], linewidth=1.4, label="WTI")

    for _, row in breakpoints.iterrows():
        ax.axvline(pd.Timestamp(row["breakpoint_date"]), linestyle="-", linewidth=1.4, alpha=0.8)

    for _, e in events.iterrows():
        event_date = pd.Timestamp(e["date"])
        ax.axvline(event_date, linestyle="--", linewidth=0.9, alpha=0.45)
        ax.text(event_date, ax.get_ylim()[1] * 0.98, str(e["event_id"]), rotation=45, fontsize=8, va="top")

    ax.set_title("结构断点与事件日期对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("WTI 美元/桶")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_structural_breaks.png", dpi=150)
    plt.close()


def plot_car_barplot(event_results: pd.DataFrame) -> None:
    if event_results.empty:
        return

    df = event_results.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(df["event_id"], df["CAR_pct"])
    ax.axhline(0, linewidth=0.8)

    for i, row in df.iterrows():
        label = str(row["significance"])
        if label:
            va = "bottom" if row["CAR_pct"] >= 0 else "top"
            ax.text(i, row["CAR_pct"], label, ha="center", va=va, fontsize=12)

    ax.set_title("各事件 CAR 对比")
    ax.set_xlabel("事件")
    ax.set_ylabel("CAR (%)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_car_barplot.png", dpi=150)
    plt.close()


def plot_return_distribution(data: pd.DataFrame, events: pd.DataFrame) -> None:
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    date_index = pd.DatetimeIndex(df["Date"])

    fig, ax = plt.subplots(figsize=(10, 6))

    returns_pct = df["WTI_return"] * 100
    ax.hist(returns_pct, bins=40, alpha=0.75, density=True)

    for _, e in events.iterrows():
        trading_date = nearest_trading_day(date_index, pd.Timestamp(e["date"]))
        if trading_date is not None:
            r = df.loc[df["Date"] == trading_date, "WTI_return"]
            if not r.empty:
                ax.axvline(r.iloc[0] * 100, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_title("WTI 日对数收益率分布")
    ax.set_xlabel("日收益率 (%)")
    ax.set_ylabel("密度")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_return_distribution.png", dpi=150)
    plt.close()


def plot_spread(data: pd.DataFrame, events: pd.DataFrame) -> None:
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["Date"], df["Brent_WTI_spread"], linewidth=1.4)

    for _, e in events.iterrows():
        ax.axvline(pd.Timestamp(e["date"]), linestyle="--", linewidth=0.8, alpha=0.5)

    ax.axhline(0, linewidth=0.8)
    ax.set_title("Brent-WTI 价差")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_spread.png", dpi=150)
    plt.close()


def plot_rolling_volatility(data: pd.DataFrame, events: pd.DataFrame, window: int = 20) -> None:
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["rolling_vol"] = df["WTI_return"].rolling(window).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["Date"], df["rolling_vol"], linewidth=1.4)

    for _, e in events.iterrows():
        ax.axvline(pd.Timestamp(e["date"]), linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(f"WTI {window}日滚动年化波动率")
    ax.set_xlabel("日期")
    ax.set_ylabel("年化波动率 (%)")
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_rolling_volatility.png", dpi=150)
    plt.close()


def plot_residual_diagnostics_for_multiple_its(
    model_ols,
    y: pd.Series,
) -> None:
    """
    Residual diagnostics for the multiple-event ITS model.

    Important:
        HAC / Newey-West changes standard errors, not fitted values or residuals.
        Therefore, residual diagnostics are based on the OLS residuals from the
        same design matrix.
    """
    fitted = model_ols.fittedvalues
    resid = model_ols.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(fitted, resid, s=15, alpha=0.7)
    axes[0, 0].axhline(0, linewidth=0.8)
    axes[0, 0].set_title("残差 vs 拟合值")
    axes[0, 0].set_xlabel("拟合值")
    axes[0, 0].set_ylabel("残差")
    axes[0, 0].grid(alpha=0.3)

    sm.qqplot(resid, line="q", ax=axes[0, 1])
    axes[0, 1].set_title("Q-Q 图")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(y.index, resid, linewidth=1.2)
    axes[1, 0].axhline(0, linewidth=0.8)
    axes[1, 0].set_title("残差时间序列")
    axes[1, 0].set_xlabel("日期")
    axes[1, 0].set_ylabel("残差")
    axes[1, 0].grid(alpha=0.3)
    plt.setp(axes[1, 0].get_xticklabels(), rotation=30, fontsize=8)

    max_lags = min(20, max(5, len(resid) // 4))
    sm.graphics.tsa.plot_acf(resid, lags=max_lags, ax=axes[1, 1])
    axes[1, 1].set_title("残差 ACF")

    plt.suptitle("ITS 残差诊断：多事件联合模型", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_residual_diagnostics.png", dpi=150)
    plt.close()


def generate_all_plots(
    data: pd.DataFrame,
    events: pd.DataFrame,
    es: EventStudy,
    event_results: pd.DataFrame,
    breakpoints: pd.DataFrame,
    model_ols,
    y_its: pd.Series,
) -> None:
    print("=" * 72)
    print("Generating figures")
    print("=" * 72)

    plot_oil_prices_with_events(data, events)
    plot_car_comparison(es)
    plot_its_analysis(data, events)
    plot_breakpoints(data, events, breakpoints)
    plot_car_barplot(event_results)
    plot_return_distribution(data, events)
    plot_spread(data, events)
    plot_rolling_volatility(data, events)
    plot_residual_diagnostics_for_multiple_its(model_ols=model_ols, y=y_its)

    print(f"Figures saved to: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    print()


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    ensure_dirs()

    data, events = build_dataset()

    es = EventStudy(data=data, events=events)
    event_results = es.run()

    fit_single_its(data=data, events=events)

    (
        its_multiple_results,
        its_model_data,
        model_hac,
        model_ols,
        y_its,
        X_its,
        dropped_cols,
    ) = fit_multiple_its(data=data, events=events)

    breakpoints = detect_breakpoints(data=data, events=events)

    generate_all_plots(
        data=data,
        events=events,
        es=es,
        event_results=event_results,
        breakpoints=breakpoints,
        model_ols=model_ols,
        y_its=y_its,
    )

    print("=" * 72)
    print("Analysis completed.")
    print("=" * 72)
    print("Output files:")
    print("  data/processed/oil_market_dataset.csv")
    print("  output/event_study_results.csv")
    print("  output/its_single_results.csv")
    print("  output/its_multiple_results.csv")
    print("  output/its_multiple_event_variables.csv")
    print("  output/breakpoints.csv")
    print("  figures/fig1_oil_price_events.png")
    print("  figures/fig2_car_comparison.png")
    print("  figures/fig3_its_analysis.png")
    print("  figures/fig4_structural_breaks.png")
    print("  figures/fig5_car_barplot.png")
    print("  figures/fig6_return_distribution.png")
    print("  figures/fig7_spread.png")
    print("  figures/fig8_rolling_volatility.png")
    print("  figures/fig9_residual_diagnostics.png")


if __name__ == "__main__":
    main()