"""
Visualization / EDA for `structured_csv_data_files/fetched_data/filtered_company_list.csv`.

This script generates Plotly HTML files into:
  `structured_csv_data_files/feature_selection_viz/filtered_company_list/`

Run with (from repo root):
  PYTHONPATH=src python3 -m data_pipelines.feature_selection.visualize_filtered_company_list

Optional overrides:
  PYTHONPATH=src python3 -m data_pipelines.feature_selection.visualize_filtered_company_list \\
    --input fetched_data/filtered_company_list.csv \\
    --out-dir feature_selection_viz/filtered_company_list
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


def _safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


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
    parser = argparse.ArgumentParser(
        description="Generate Plotly HTML EDA for filtered_company_list.csv"
    )
    parser.add_argument(
        "--input",
        default="fetched_data/filtered_company_list.csv",
        help="Path to CSV (absolute, or relative to structured_csv_data_files/).",
    )
    parser.add_argument(
        "--out-dir",
        default="feature_selection_viz/filtered_company_list",
        help="Output directory (absolute, or relative to structured_csv_data_files/).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    out_base = _resolve_out_dir(args.out_dir)

    df = pd.read_csv(input_path)
    print("Loaded filtered_company_list.csv")
    print("shape:", df.shape)

    out_base.mkdir(parents=True, exist_ok=True)

    # ---------------------------- Missingness ----------------------------
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    missing_top = missing_pct.head(15).to_frame("missing_pct")
    print("Top missing columns (percent):")
    print(missing_top.to_string())

    fig_missing = go.Figure(
        data=[
            go.Bar(
                x=missing_top.index.tolist(),
                y=missing_top["missing_pct"].tolist(),
                marker=dict(color="rgba(255, 127, 14, 0.75)"),
            )
        ]
    )
    fig_missing.update_layout(
        title="Top Missing Columns (filtered_company_list.csv)",
        xaxis_title="Column",
        yaxis_title="Missing (%)",
        template="plotly_white",
    )
    _to_html(fig_missing, out_base / "missing_values_top15.html")

    # ---------------------------- Numeric distributions ----------------------------
    numeric_cols = _safe_numeric_cols(df)
    numeric_cols = [c for c in numeric_cols if c not in {"Year"}]

    if numeric_cols:
        # Histogram per numeric column (limit to keep things readable).
        for c in numeric_cols[:8]:
            fig_hist = go.Figure(data=[go.Histogram(x=df[c].dropna(), nbinsx=40)])
            fig_hist.update_layout(
                title="Distribution: " + c,
                xaxis_title=c,
                yaxis_title="Count",
                template="plotly_white",
            )
            _to_html(fig_hist, out_base / ("hist_" + c + ".html"))

    # ---------------------------- Categorical counts ----------------------------
    categorical_preferred = ["Exchange", "Sector", "Industry", "Ticker"]
    categorical_cols = [c for c in categorical_preferred if c in df.columns]

    # For 'Ticker' keep it out of category counts (too many).
    for c in categorical_cols:
        if c == "Ticker":
            continue
        counts = df[c].value_counts().head(20)
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.astype(str).tolist(),
                    y=counts.values.tolist(),
                    marker=dict(color="rgba(31, 119, 180, 0.75)"),
                )
            ]
        )
        fig_bar.update_layout(
            title="Top 20 Counts: " + c,
            xaxis_title=c,
            yaxis_title="Count",
            template="plotly_white",
        )
        _to_html(fig_bar, out_base / ("count_top20_" + c + ".html"))

    # ---------------------------- MarketCap vs ProfitMargin ----------------------------
    if "MarketCap" in df.columns and "ProfitMargin" in df.columns:
        plot_df = df[["MarketCap", "ProfitMargin"]].dropna()
        # Avoid too many points (still small usually, but keep safe).
        if len(plot_df) > 5000:
            plot_df = plot_df.sample(n=5000, random_state=42)

        fig_scatter = go.Figure(
            data=[
                go.Scatter(
                    x=plot_df["MarketCap"],
                    y=plot_df["ProfitMargin"],
                    mode="markers",
                    marker=dict(size=7, opacity=0.5),
                )
            ]
        )
        fig_scatter.update_layout(
            title="MarketCap vs ProfitMargin",
            xaxis_title="MarketCap",
            yaxis_title="ProfitMargin",
            template="plotly_white",
        )
        _to_html(fig_scatter, out_base / "scatter_marketcap_vs_profitmargin.html")

    # ---------------------------- Top tickers by MarketCap ----------------------------
    if "MarketCap" in df.columns and "Ticker" in df.columns:
        top = df[["Ticker", "MarketCap"]].dropna().sort_values("MarketCap", ascending=False).head(10)
        fig_top = go.Figure(
            data=[
                go.Bar(
                    x=top["Ticker"].astype(str).tolist(),
                    y=top["MarketCap"].tolist(),
                    marker=dict(color="rgba(44, 160, 44, 0.75)"),
                )
            ]
        )
        fig_top.update_layout(
            title="Top 10 Tickers by MarketCap",
            xaxis_title="Ticker",
            yaxis_title="MarketCap",
            template="plotly_white",
        )
        _to_html(fig_top, out_base / "top10_marketcap_tickers.html")

    # ---------------------------- Compact table (top missing) ----------------------------
    overview_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_pct": [(df[c].isna().mean() * 100.0) for c in df.columns],
        }
    ).sort_values("missing_pct", ascending=False)

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
                        [round(float(v), 4) for v in overview_df["missing_pct"].tolist()],
                    ],
                    fill_color="white",
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(title="filtered_company_list.csv Column Overview (Top 30 Missing)")
    _to_html(fig_table, out_base / "column_overview_top30_missing.html")

    print("Visualization files written to:")
    print(out_base)


if __name__ == "__main__":
    main()

