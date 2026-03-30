"""
EDA / visualization for `structured_csv_data_files/fetched_data/Weekend_dataset.csv`.

Adds plots tailored to weekend-gap modeling:
  - Ticker-level empirical frequencies (P(gap up), mean gap, sample size)
  - Conditional gap metrics by quantile bins of key Friday-row features
  - Correlation ranking vs weekend_gap_pct (Friday features)

Outputs Plotly HTML under:
  `structured_csv_data_files/feature_selection_viz/weekend_dataset/`

Run from repo root:
  PYTHONPATH=src python3 -m data_pipelines.feature_selection.visualize_weekend_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DATA_DIR
from data_pipelines.feature_selection.visualize_dataset import (
    _build_weekend_gap_pairs,
    _resolve_input_path,
    _resolve_out_dir,
    _to_html,
    _top_correlated_features,
)


def _merge_friday_features(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    base = base.dropna(subset=["Date", "Ticker"])
    base = base.rename(columns={"Date": "friday_date"})
    return pairs.merge(base, on=["Ticker", "friday_date"], how="left", suffixes=("", "_fridayrow"))


def _ticker_weekend_stats(pairs: pd.DataFrame, min_weekends: int = 40) -> pd.DataFrame:
    """Per-ticker empirical gap stats (requires weekend_gap_pct)."""
    g = pairs.groupby("Ticker", sort=False)
    out = g.agg(
        n_weekends=("weekend_gap_pct", "count"),
        mean_gap_pct=("weekend_gap_pct", "mean"),
        std_gap_pct=("weekend_gap_pct", "std"),
        p_gap_up=("weekend_gap_pct", lambda s: float((s > 0).mean()) if len(s) else np.nan),
        p_gap_gt_half_pct=("weekend_gap_pct", lambda s: float((s.abs() > 0.005).mean()) if len(s) else np.nan),
    ).reset_index()
    out = out[out["n_weekends"] >= min_weekends]
    return out.sort_values("n_weekends", ascending=False)


def _plot_ticker_p_up(stats: pd.DataFrame, out_path: Path, top_n: int = 35) -> None:
    top = stats.nlargest(top_n, "n_weekends")
    fig = go.Figure(
        data=[
            go.Bar(
                y=top["Ticker"].astype(str).tolist()[::-1],
                x=top["p_gap_up"].tolist()[::-1],
                orientation="h",
                marker=dict(color=top["mean_gap_pct"].tolist()[::-1], colorscale="RdBu", cmid=0),
                text=[f"n={int(n)}" for n in top["n_weekends"].tolist()[::-1]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"Empirical P(gap > 0) by ticker (top {top_n} by sample size; color = mean gap %)",
        xaxis_title="P(Mon open > Fri close)",
        yaxis_title="Ticker",
        template="plotly_white",
        height=900,
    )
    _to_html(fig, out_path)


def _plot_ticker_mean_gap_hist(stats: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure(data=[go.Histogram(x=stats["mean_gap_pct"].dropna(), nbinsx=40)])
    fig.update_layout(
        title="Across tickers: distribution of mean weekend_gap_pct",
        xaxis_title="Mean weekend gap % (per ticker)",
        yaxis_title="Tickers",
        template="plotly_white",
    )
    _to_html(fig, out_path)


def _conditional_decile_table(
    merged: pd.DataFrame,
    feat: str,
    target: str = "weekend_gap_pct",
    n_bins: int = 10,
) -> pd.DataFrame | None:
    if feat not in merged.columns or target not in merged.columns:
        return None
    d = merged[[feat, target]].dropna()
    if len(d) < n_bins * 20:
        return None
    try:
        d["bin"] = pd.qcut(d[feat], q=n_bins, duplicates="drop")
    except ValueError:
        return None
    g = d.groupby("bin", observed=True)
    rows = []
    for bin_label, sub in g:
        rows.append(
            {
                "bin": str(bin_label),
                "n": len(sub),
                f"{feat}_mean": float(sub[feat].mean()),
                f"{target}_mean": float(sub[target].mean()),
                "p_gap_up": float((sub[target] > 0).mean()),
                "p_gap_large": float((sub[target].abs() > 0.01).mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_conditional_deciles(
    df_dec: pd.DataFrame,
    feat: str,
    out_path: Path,
    title_suffix: str = "",
    target: str = "weekend_gap_pct",
) -> None:
    if df_dec is None or df_dec.empty:
        return
    mean_col = f"{target}_mean"
    if mean_col not in df_dec.columns:
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df_dec["bin"], y=df_dec["p_gap_up"], name="P(gap > 0)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_dec["bin"],
            y=df_dec[mean_col],
            mode="lines+markers",
            name="Mean gap %",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Conditional metrics by {feat} decile {title_suffix}",
        template="plotly_white",
        barmode="group",
        xaxis_title=f"{feat} bin (quantile)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="P(gap > 0)", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Mean weekend gap %", secondary_y=True)
    _to_html(fig, out_path)


def _plot_correlation_rank(
    merged: pd.DataFrame,
    target: str,
    out_path: Path,
    top_n: int = 18,
    exclude: set[str] | None = None,
) -> None:
    exclude = exclude or set()
    exclude |= {target, "days_between", "monday_open", "friday_close"}
    numeric = merged.select_dtypes(include="number")
    cols = [c for c in numeric.columns if c not in exclude]
    if target not in numeric.columns:
        return
    use_cols = list(dict.fromkeys([c for c in cols if c != target] + [target]))
    corr = numeric[use_cols].corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr = corr.dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)
    fig = go.Figure(
        data=[
            go.Bar(
                x=corr.index.astype(str).tolist(),
                y=corr.values.tolist(),
                marker=dict(
                    color=[
                        "rgba(214, 39, 40, 0.75)" if v < 0 else "rgba(31, 119, 180, 0.75)"
                        for v in corr.values
                    ]
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Pearson correlation with {target} (Friday-row features)",
        xaxis_title="Feature",
        yaxis_title="Correlation",
        template="plotly_white",
    )
    _to_html(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekend_dataset EDA plots (Mon/Fri CSV).")
    parser.add_argument(
        "--input",
        default="fetched_data/Weekend_dataset.csv",
        help="Path to Weekend_dataset.csv (relative to structured_csv_data_files/ or absolute).",
    )
    parser.add_argument(
        "--out-dir",
        default="feature_selection_viz/weekend_dataset",
        help="Output directory (relative to structured_csv_data_files/ or absolute).",
    )
    parser.add_argument(
        "--min-ticker-weekends",
        type=int,
        default=40,
        help="Minimum paired weekends required to include a ticker in ticker-level charts.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    out_base = _resolve_out_dir(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print("Loaded:", input_path.name)
    print("shape:", df.shape)

    pairs = _build_weekend_gap_pairs(df)
    if pairs.empty:
        print("No weekend pairs; check Date/Open/Close columns.")
        return

    merged = _merge_friday_features(df, pairs)
    merged = merged.dropna(subset=["weekend_gap_pct"])

    # --- Ticker empirical probabilities ---
    stats = _ticker_weekend_stats(pairs, min_weekends=args.min_ticker_weekends)
    if not stats.empty:
        _plot_ticker_p_up(stats, out_base / "weekend_ticker_p_gap_up_top35.html")
        _plot_ticker_mean_gap_hist(stats, out_base / "weekend_ticker_mean_gap_pct_hist.html")

        fig_sc = go.Figure(
            data=[
                go.Scatter(
                    x=stats["std_gap_pct"],
                    y=stats["mean_gap_pct"],
                    mode="markers",
                    text=stats["Ticker"],
                    marker=dict(
                        size=8,
                        opacity=0.65,
                        color=stats["p_gap_up"],
                        colorscale="Viridis",
                        showscale=True,
                    ),
                )
            ]
        )
        fig_sc.update_layout(
            title="Ticker-level: std vs mean weekend gap % (color = P(gap > 0))",
            xaxis_title="Std dev of weekend gap %",
            yaxis_title="Mean weekend gap %",
            template="plotly_white",
        )
        _to_html(fig_sc, out_base / "weekend_ticker_mean_vs_std_gap_scatter.html")

    # --- Important features: correlation + decile conditionals ---
    _plot_correlation_rank(
        merged,
        "weekend_gap_pct",
        out_base / "weekend_gap_friday_feature_correlations_ranked.html",
    )

    top_corr = _top_correlated_features(merged, "weekend_gap_pct", top_n=12)
    if not top_corr.empty:
        fig_top = go.Figure(
            data=[
                go.Bar(
                    x=top_corr.index.tolist(),
                    y=top_corr.values.tolist(),
                    marker=dict(color="rgba(44, 160, 44, 0.8)"),
                )
            ]
        )
        fig_top.update_layout(
            title="Top |correlation| features vs weekend_gap_pct (numeric columns)",
            xaxis_title="Feature",
            yaxis_title="Correlation",
            template="plotly_white",
        )
        _to_html(fig_top, out_base / "weekend_gap_top_correlated_features_bar.html")

    # Prefer these for interpretability; fall back if missing.
    decile_features = [
        "VolumeRatio",
        "FridayPosition",
        "FiveDStdDev",
        "WeeklyReturn",
        "CloseVSma20",
        "IntraWeekVolatility",
        "Volume",
    ]
    for feat in decile_features:
        if feat not in merged.columns:
            continue
        tbl = _conditional_decile_table(merged, feat, n_bins=10)
        if tbl is None or tbl.empty:
            continue
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in feat)
        _plot_conditional_deciles(tbl, feat, out_base / f"weekend_conditional_gap_by_{safe}_decile.html")

    # Dollar volume proxy if we have OHLC
    if {"Close", "Volume"}.issubset(merged.columns):
        merged = merged.copy()
        merged["dollar_volume"] = merged["Close"] * merged["Volume"]
        tbl = _conditional_decile_table(merged, "dollar_volume", n_bins=10)
        if tbl is not None and not tbl.empty:
            _plot_conditional_deciles(
                tbl,
                "dollar_volume",
                out_base / "weekend_conditional_gap_by_dollar_volume_decile.html",
            )

    print("Wrote weekend visualization files to:")
    print(out_base)


if __name__ == "__main__":
    main()

