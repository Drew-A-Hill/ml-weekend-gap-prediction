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
        "price_aggregates": {
            # "weekly_return": pa.weekly_return,
            # "intra_week_volatility": pa.intra_week_volatility,
            # "weekly_range": pa.weekly_range,
            # "friday_position": pa.friday_position,
            # "open_close_spread": pa.open_close_spread,
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

        # "volume_indicators": {
        #     "obv": vol.obv,
        #     "mfi": vol.mfi,
        #     "volume_ratio": vol.volume_ratio,
        # },

        "fundamental_indicators": {
            "gross_margin": fi.gross_margin,
            # "operating_margin": fi.operating_margin,
            "net_margin": fi.net_margin,
            # "debt_to_equity_ratio": fi.debt_to_equity_ratio,
            "roa": fi.roa,
            "rev_growth_qoq": fi.rev_growth_qoq,
        },
    }

def add_indicators(data: pd.DataFrame, indicators: list[str] | None=None, add_all: bool = False) -> pd.DataFrame:
    """

    """
    df = data.copy()
    action_map = _indicator_map()

    if add_all:
        for key, val in action_map.items():
            for t, v in val.items():
                df = v(df)

        return df

    if indicators:
        for key, val in action_map.items():
            for t, v in val.items():
                if t in indicators:
                    df = v(df)

    return df



