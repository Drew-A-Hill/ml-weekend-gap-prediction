"""

"""
from typing import Any

import pandas as pd

import data_pipelines.api_data_ingestion.indicator_calcs.intermediate_calcs as ic
import data_pipelines.api_data_ingestion.indicator_calcs.price_aggregate as pa
import data_pipelines.api_data_ingestion.indicator_calcs.momentum as mi
import data_pipelines.api_data_ingestion.indicator_calcs.trend as tr
import data_pipelines.api_data_ingestion.indicator_calcs.volatility as vi
import data_pipelines.api_data_ingestion.indicator_calcs.volume as vol
import data_pipelines.api_data_ingestion.indicator_calcs.fundamental as fi

def _indicator_map() -> dict[str, dict[str, Any]]:
    return {
        "intermediate_price": {
            "daily_return": ic.daily_returns,
            "monday_open": ic.monday_open,
            "friday_open": ic.friday_open,
            "friday_close": ic.friday_close,
            "friday_volume": ic.friday_volume,
            "weekly_high": ic.weekly_high,
            "weekly_low": ic.weekly_low,
            "weekly_avg_volume": ic.weekly_avg_volume,
            "prev_friday_close": ic.prev_friday_close,
            "price_change": ic.price_change,
            "gain": ic.gain,
            "loss": ic.loss,
            "avg_gain": ic.avg_gain,
            "avg_loss": ic.avg_loss,
            "ema_12": ic.ema_12,
            "ema_26": ic.ema_26,
            "ema_50": ic.ema_50,
            "sma_20": ic.sma_20,
            "rolling_std_n": ic.rolling_std_n,
            "upper_band": ic.upper_band,
            "lower_band": ic.lower_band,
            "prev_close": ic.prev_close,
            "tr": ic.tr,
            "typical_price": ic.typical_price,
            "raw_money_flow": ic.raw_money_flow,
            "prev_quarter_revenue": ic.prev_quarter_revenue,
        },

        "price_aggregates": {
            "weekly_return": pa.weekly_return,
            "intra_week_volatility": pa.intra_week_volatility,
            "weekly_range": pa.weekly_range,
            "friday_position": pa.friday_position,
            "open_close_spread": pa.open_close_spread,
        },

        "momentum_indicators": {
            "rsi": mi.rsi,
            "macd": mi.macd,
            "roc": mi.roc,
            "stoch_perc_k": mi.stoch_perc_k,
        },

        "trend_indicators": {
            "close_v_ema50": tr.close_v_ema50,
            "close_v_sma20": tr.close_v_sma20,
            "adx": tr.adx,
        },

        "volatility_indicators": {
            "bollinger_band_width": vi.bollinger_band_width,
            "atr": vi.atr,
            "five_d_std_dev": vi.five_d_std_dev,
        },

        "volume_indicators": {
            "obv": vol.obv,
            "mfi": vol.mfi,
            "volume_ratio": vol.volume_ratio,
        },

        "fundamental_indicators": {
            "gross_margin": fi.gross_margin,
            "operating_margin": fi.operating_margin,
            "net_margin": fi.net_margin,
            "debt_to_equity_ratio": fi.debt_to_equity_ratio,
            "roa": fi.roa,
            "rev_growth_qoq": fi.rev_growth_qoq,
        },
    }

def add_indicators(data: pd.DataFrame, indicators: list[str] | None=None, add_all: bool = False) -> pd.DataFrame:
    """

    """
    df: pd.DataFrame = data.copy()

    selected = set(indicators or [])

    for group_dict in _indicator_map().values():
        for indicator_name, indicator_fn in group_dict.items():
            if add_all or indicator_name in selected:
                result = indicator_fn(df)

                if isinstance(result, pd.Series):
                    df[indicator_name] = result
                else:
                    df = result

    return df








