"""
Visualization / EDA for `structured_csv_data_files/fetched_data/dataset.csv`.

This script generates Plotly HTML files into:
  `structured_csv_data_files/feature_selection_viz/dataset/`

Run with (from repo root):
  PYTHONPATH=src python3 -m data_pipelines.feature_selection.visualize_dataset

Optional overrides:
  PYTHONPATH=src python3 -m data_pipelines.feature_selection.visualize_dataset \\
    --input fetched_data/dataset.csv \\
    --out-dir feature_selection_viz/dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from config import DATA_DIR


def _to_html(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _top_missing(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    return missing_pct.head(top_n).to_frame("missing_pct")


def _pick_target_column(df: pd.DataFrame) -> str | None:
    preferred = ["open_close_spread", "weekend_gap", "weekly_return", "gap"]
    for col in preferred:
        if col in df.columns:
            return col

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if numeric_cols:
        return numeric_cols[0]
    return None


def _build_weekend_gap_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (Ticker, Friday -> Monday) pairs and compute weekend gap metrics.

    Requires columns: Date, Ticker, Close, Open
    """
    required = {"Date", "Ticker", "Close", "Open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing required columns for weekend-gap viz: {sorted(missing)}")

    dfx = df[["Date", "Ticker", "Close", "Open"]].copy()
    # Force UTC to handle mixed offsets safely, then drop timezone to keep .dt ops simple.
    dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    dfx = dfx.dropna(subset=["Date", "Ticker"])
    dfx = dfx.sort_values(["Ticker", "Date"])

    dfx["weekday"] = dfx["Date"].dt.weekday  # Monday=0 ... Friday=4

    fridays = dfx[dfx["weekday"] == 4][["Ticker", "Date", "Close"]].rename(
        columns={"Date": "friday_date", "Close": "friday_close"}
    )
    mondays = dfx[dfx["weekday"] == 0][["Ticker", "Date", "Open"]].rename(
        columns={"Date": "monday_date", "Open": "monday_open"}
    )

    if fridays.empty or mondays.empty:
        return pd.DataFrame()

    # NOTE: pandas `merge_asof(..., by="Ticker")` still requires the left key to be
    # globally sorted; to avoid "left keys must be sorted" edge cases, do it per-ticker.
    all_pairs: list[pd.DataFrame] = []
    for ticker, monday_grp in mondays.groupby("Ticker", sort=False):
        friday_grp = fridays[fridays["Ticker"] == ticker]
        if friday_grp.empty:
            continue

        monday_grp = monday_grp.sort_values("monday_date")
        friday_grp = friday_grp.sort_values("friday_date")

        pairs_t = pd.merge_asof(
            monday_grp,
            friday_grp,
            left_on="monday_date",
            right_on="friday_date",
            direction="backward",
            tolerance=pd.Timedelta(days=3),
        )
        pairs_t["Ticker"] = ticker
        all_pairs.append(pairs_t)

    if not all_pairs:
        return pd.DataFrame()

    pairs = pd.concat(all_pairs, ignore_index=True)

    pairs = pairs.dropna(subset=["friday_date", "friday_close", "monday_open"])
    pairs["days_between"] = (pairs["monday_date"] - pairs["friday_date"]).dt.days
    pairs = pairs[(pairs["days_between"] >= 2) & (pairs["days_between"] <= 3)]

    # Gap metrics
    pairs["weekend_gap_abs"] = pairs["monday_open"] - pairs["friday_close"]
    pairs["weekend_gap_pct"] = pairs["weekend_gap_abs"] / pairs["friday_close"]

    return pairs.reset_index(drop=True)


def _top_correlated_features(df: pd.DataFrame, target_col: str, top_n: int = 8) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    if target_col not in numeric.columns:
        return pd.Series(dtype=float)

    corr = numeric.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    corr = corr.dropna()
    if corr.empty:
        return corr
    return corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)


def _resolve_input_path(input_arg: str) -> Path:
    p = Path(input_arg)
    if p.is_absolute():
        return p
    return DATA_DIR / p


def _resolve_out_dir(out_dir_arg: str) -> Path:
    p = Path(out_dir_arg)
    if p.is_absolute():
        return p
    return DATA_DIR / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Plotly HTML EDA for dataset.csv")
    parser.add_argument(
        "--input",
        default="fetched_data/dataset.csv",
        help="Path to dataset CSV (absolute, or relative to structured_csv_data_files/).",
    )
    parser.add_argument(
        "--out-dir",
        default="feature_selection_viz/dataset",
        help="Output directory (absolute, or relative to structured_csv_data_files/).",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=8000,
        help="Max points to plot in scatter plots (for readability/perf).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    out_base = _resolve_out_dir(args.out_dir)

    df = pd.read_csv(input_path)

    print("Loaded dataset.csv")
    print("shape:", df.shape)

    out_base.mkdir(parents=True, exist_ok=True)

    # ---------------------------- Missingness overview ----------------------------
    missing_top = _top_missing(df, top_n=15)
    print("Top missing columns (percent):")
    print(missing_top.to_string())

    fig_missing = go.Figure(
        data=[
            go.Bar(
                x=missing_top.index.tolist(),
                y=missing_top["missing_pct"].tolist(),
                marker=dict(color="rgba(31, 119, 180, 0.75)"),
            )
        ]
    )
    fig_missing.update_layout(
        title="Top Missing Columns",
        xaxis_title="Column",
        yaxis_title="Missing (%)",
        template="plotly_white",
    )
    _to_html(fig_missing, out_base / "missing_values_top15.html")

    # ---------------------------- Weekend gap (Friday close -> Monday open) ----------------------------
    try:
        weekend_pairs = _build_weekend_gap_pairs(df)
    except Exception as e:
        weekend_pairs = pd.DataFrame()
        print("Weekend-gap pairing failed:", str(e))

    if not weekend_pairs.empty:
        # Friday Close vs Monday Open scatter + y=x reference
        plot_pairs = weekend_pairs.copy()
        if len(plot_pairs) > args.max_scatter_points:
            plot_pairs = plot_pairs.sample(n=args.max_scatter_points, random_state=42)

        fig_fc_mo = go.Figure()
        fig_fc_mo.add_trace(
            go.Scatter(
                x=plot_pairs["friday_close"],
                y=plot_pairs["monday_open"],
                mode="markers",
                marker=dict(size=6, opacity=0.55),
                name="Pairs",
                text=plot_pairs["Ticker"],
            )
        )
        min_xy = float(min(plot_pairs["friday_close"].min(), plot_pairs["monday_open"].min()))
        max_xy = float(max(plot_pairs["friday_close"].max(), plot_pairs["monday_open"].max()))
        fig_fc_mo.add_trace(
            go.Scatter(
                x=[min_xy, max_xy],
                y=[min_xy, max_xy],
                mode="lines",
                line=dict(dash="dash", color="grey"),
                name="y = x",
            )
        )
        fig_fc_mo.update_layout(
            title="Weekend Gap Pairs: Friday Close vs Monday Open",
            xaxis_title="Friday Close",
            yaxis_title="Monday Open",
            template="plotly_white",
        )
        _to_html(fig_fc_mo, out_base / "weekend_gap_friday_close_vs_monday_open.html")

        # Gap % distribution
        fig_gap = go.Figure(
            data=[go.Histogram(x=weekend_pairs["weekend_gap_pct"].dropna(), nbinsx=80)]
        )
        fig_gap.update_layout(
            title="Weekend Gap % Distribution ((Mon Open - Fri Close) / Fri Close)",
            xaxis_title="Weekend Gap (%)",
            yaxis_title="Count",
            template="plotly_white",
        )
        _to_html(fig_gap, out_base / "weekend_gap_pct_hist.html")

        # Top features vs weekend gap % (based on correlation within weekend pairs)
        # Attach extra numeric features from the original DF by joining on the friday_date row.
        base = df.copy()
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        base = base.dropna(subset=["Date", "Ticker"])
        base = base.rename(columns={"Date": "friday_date"})

        # Merge features on (Ticker, friday_date)
        merged = weekend_pairs.merge(base, on=["Ticker", "friday_date"], how="left", suffixes=("", "_fridayrow"))

        top_corr = _top_correlated_features(merged, "weekend_gap_pct", top_n=10)
        if not top_corr.empty:
            fig_top = go.Figure(
                data=[
                    go.Bar(
                        x=top_corr.index.tolist(),
                        y=top_corr.values.tolist(),
                        marker=dict(color="rgba(214, 39, 40, 0.75)"),
                    )
                ]
            )
            fig_top.update_layout(
                title="Top Correlated Features vs Weekend Gap (%) (Friday-row features)",
                xaxis_title="Feature",
                yaxis_title="Correlation with weekend_gap_pct",
                template="plotly_white",
            )
            _to_html(fig_top, out_base / "weekend_gap_top_feature_correlations.html")

            # Scatter plots for the top few features
            for feat in top_corr.index.tolist()[:6]:
                if feat not in merged.columns:
                    continue
                plot_df = merged[[feat, "weekend_gap_pct"]].dropna()
                if plot_df.empty:
                    continue
                if len(plot_df) > args.max_scatter_points:
                    plot_df = plot_df.sample(n=args.max_scatter_points, random_state=42)

                fig_sc = go.Figure(
                    data=[
                        go.Scatter(
                            x=plot_df[feat],
                            y=plot_df["weekend_gap_pct"],
                            mode="markers",
                            marker=dict(size=6, opacity=0.5),
                        )
                    ]
                )
                fig_sc.update_layout(
                    title=f"Weekend Gap (%) vs {feat}",
                    xaxis_title=feat,
                    yaxis_title="weekend_gap_pct",
                    template="plotly_white",
                )
                safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in feat)
                _to_html(fig_sc, out_base / f"weekend_gap_scatter_vs_{safe_name}.html")

    # ---------------------------- Basic distributions (fallback / general) ----------------------------
    target_col = _pick_target_column(df)
    if target_col is None:
        print("No numeric columns found; skipping numeric visualizations.")
        return

    print("Using target column for plots:", target_col)

    # Histogram of target
    fig_hist = go.Figure(data=[go.Histogram(x=df[target_col].dropna(), nbinsx=50)])
    fig_hist.update_layout(
        title="Distribution: " + target_col,
        xaxis_title=target_col,
        yaxis_title="Count",
        template="plotly_white",
    )
    _to_html(fig_hist, out_base / "target_hist.html")

    # Boxplot by Quarter (if present)
    if "Quarter" in df.columns:
        dfx = df[["Quarter", target_col]].dropna()
        fig_box = go.Figure(
            data=[go.Box(x=dfx["Quarter"].astype(str), y=dfx[target_col], boxpoints=False)]
        )
        fig_box.update_layout(
            title="Target by Quarter: " + target_col,
            xaxis_title="Quarter",
            yaxis_title=target_col,
            template="plotly_white",
        )
        _to_html(fig_box, out_base / "target_box_by_quarter.html")

    # ---------------------------- Ticker coverage ----------------------------
    if "Ticker" in df.columns:
        counts = df["Ticker"].value_counts()
        top_tickers = counts.head(10)
        fig_ticker = go.Figure(
            data=[
                go.Bar(
                    x=top_tickers.index.tolist(),
                    y=top_tickers.values.tolist(),
                    marker=dict(color="rgba(44, 160, 44, 0.75)"),
                )
            ]
        )
        fig_ticker.update_layout(
            title="Top 10 Tickers by Row Count",
            xaxis_title="Ticker",
            yaxis_title="Rows",
            template="plotly_white",
        )
        _to_html(fig_ticker, out_base / "ticker_rows_top10.html")

    # ---------------------------- Correlation heatmap ----------------------------
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        numeric_missing_pct = (numeric_df.isna().mean() * 100.0).sort_values(ascending=True)
        keep_cols = numeric_missing_pct[numeric_missing_pct < 50].index.tolist()
        # Keep the heatmap readable.
        keep_cols = keep_cols[:20]

        if len(keep_cols) >= 2:
            corr = numeric_df[keep_cols].corr()

            fig_corr = go.Figure(
                data=[
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale="RdBu",
                        reversescale=True,
                        zmin=-1,
                        zmax=1,
                    )
                ]
            )
            fig_corr.update_layout(
                title="Correlation Heatmap (Top Non-Missing Numeric Columns)",
                template="plotly_white",
            )
            _to_html(fig_corr, out_base / "correlation_heatmap_top20.html")

    # ---------------------------- Feature scatter (target vs a likely feature) ----------------------------
    likely_x_candidates = ["friday_position", "open_close_spread", "weekly_return", "intra_week_volatility"]
    x_col = None
    for c in likely_x_candidates:
        if c in df.columns and c != target_col:
            x_col = c
            break

    if x_col is not None:
        dfx = df[[x_col, target_col]].dropna()
        # Avoid plotting too many points.
        if len(dfx) > args.max_scatter_points:
            dfx = dfx.sample(n=args.max_scatter_points, random_state=42)

        fig_scatter = go.Figure(
            data=[
                go.Scatter(
                    x=dfx[x_col],
                    y=dfx[target_col],
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                )
            ]
        )
        fig_scatter.update_layout(
            title="Scatter: " + x_col + " vs " + target_col,
            xaxis_title=x_col,
            yaxis_title=target_col,
            template="plotly_white",
        )
        _to_html(fig_scatter, out_base / "scatter_target_vs_feature.html")

    # ---------------------------- Compact overview table ----------------------------
    overview_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = float(df[col].isna().mean() * 100.0)
        overview_rows.append((col, dtype, missing_pct))

    # Keep HTML table readable.
    overview_df = pd.DataFrame(overview_rows, columns=["column", "dtype", "missing_pct"]).sort_values(
        "missing_pct", ascending=False
    )
    overview_df = overview_df.head(30)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Column", "Dtype", "Missing (%)"],
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[
                        overview_df["column"].tolist(),
                        overview_df["dtype"].tolist(),
                        [round(v, 4) for v in overview_df["missing_pct"].tolist()],
                    ],
                    fill_color="white",
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(title="dataset.csv Column Overview (Top 30 Missing)")
    _to_html(fig_table, out_base / "column_overview_top30_missing.html")

    print("Visualization files written to:")
    print(out_base)


if __name__ == "__main__":
    main()

