"""
JAI Trading Research — Streamlit web dashboard.

Run with:
    uv run streamlit run app.py

Shares all data logic with scripts/daily_report.py via
src/dashboard/data.build_dashboard_data().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.data import DashboardData, build_dashboard_data

# ─────────────────────────────────────────────────────────────────────────────
#  Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JAI Trading Research",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Cached data loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading market data…")
def load_data(refresh: bool = False) -> DashboardData:
    return build_dashboard_data(refresh=refresh, ml_threshold=7.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Column configs
# ─────────────────────────────────────────────────────────────────────────────

_SCREENER_COL_CONFIG = {
    "symbol":         st.column_config.TextColumn("Symbol", width="small"),
    "current_price":  st.column_config.NumberColumn("Price", format="$%.2f", width="small"),
    "change_pct":     st.column_config.NumberColumn("Change %", format="%.1f%%", width="small"),
    "volume_ratio":   st.column_config.NumberColumn("Vol Ratio", format="%.1fx", width="small"),
    "rsi":            st.column_config.NumberColumn("RSI", format="%.0f", width="small"),
    "category_score": st.column_config.NumberColumn("Score", format="%.0f", width="small"),
}

_ALL_SCORES_COL_CONFIG = {
    "symbol":       st.column_config.TextColumn("Symbol"),
    "score":        st.column_config.ProgressColumn("Score", min_value=0, max_value=10,
                                                     format="%.1f"),
    "rsi":          st.column_config.NumberColumn("RSI", format="%.0f"),
    "volume_ratio": st.column_config.NumberColumn("Vol Ratio", format="%.1fx"),
    "has_ml":       st.column_config.CheckboxColumn("ML Model"),
    "ml_proba":     st.column_config.NumberColumn("ML Prob", format="%.0f%%"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Table / chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_screener_table(df: pd.DataFrame, empty_msg: str) -> None:
    if df.empty:
        st.info(empty_msg)
        return
    display = df[[c for c in _SCREENER_COL_CONFIG if c in df.columns]].copy()
    if "ml_proba" in display.columns:
        display["ml_proba"] = display["ml_proba"].apply(
            lambda x: x * 100 if x is not None and not (isinstance(x, float) and np.isnan(x)) else None
        )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=_SCREENER_COL_CONFIG,
    )


def _build_price_chart(symbol: str, prices_dict: dict[str, pd.DataFrame]) -> go.Figure:
    df = prices_dict.get(symbol)
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False,
                           font=dict(size=16), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    tail = df.tail(90).copy()
    tail["sma20"] = tail["close"].rolling(20).mean()
    tail["sma50"] = tail["close"].rolling(50, min_periods=20).mean()
    dates = tail.index

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    fig.add_trace(go.Scatter(
        x=dates, y=tail["close"],
        name="Close",
        line=dict(color="#4C9BE8", width=2),
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=tail["sma20"],
        name="SMA 20",
        line=dict(color="#F0A500", width=1.2, dash="dot"),
        hovertemplate="SMA20: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=tail["sma50"],
        name="SMA 50",
        line=dict(color="#9B59B6", width=1.2, dash="dot"),
        hovertemplate="SMA50: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    colors = [
        "#26A69A" if c >= o else "#EF5350"
        for c, o in zip(tail["close"], tail["open"])
    ]
    fig.add_trace(go.Bar(
        x=dates, y=tail["volume"],
        name="Volume",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.7,
        hovertemplate="%{x|%b %d}<br>Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    today = dates[-1]
    fig.add_vline(
        x=today, line_width=1, line_dash="dash",
        line_color="rgba(255,255,255,0.35)", row="all", col=1,
    )

    fig.update_layout(
        title=dict(text=f"{symbol} — last 90 days", font=dict(size=15)),
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
                   tickprefix="$"),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        bargap=0.15,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Signal card helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trade_link(symbol: str, asset_class: str) -> str:
    if asset_class == "crypto":
        url_sym = symbol.replace("/", "_")
        return f"[Trade on Binance](https://www.binance.com/en/trade/{url_sym}?type=spot)"
    return f"[View on TradingView](https://www.tradingview.com/symbols/NASDAQ-{symbol}/)"


def _render_signal_card(s: dict, expanded: bool = True) -> None:
    sym       = s["symbol"]
    score     = s.get("score", 0)
    has_ml    = s.get("has_ml", False)
    proba     = s.get("ml_proba")
    rsi       = s.get("rsi", float("nan"))
    vol       = s.get("volume_ratio", float("nan"))
    asset_cls = s.get("asset_class", "stock")
    label     = "ML + technical" if has_ml else "technical-only"

    with st.expander(f"**{sym}**  —  Score {score:.1f}/10  ({label})", expanded=expanded):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Score", f"{score:.1f}/10")
        mc2.metric("RSI", f"{rsi:.0f}" if not (isinstance(rsi, float) and np.isnan(rsi)) else "n/a")
        mc3.metric("Vol Ratio", f"{vol:.1f}×" if not (isinstance(vol, float) and np.isnan(vol)) else "n/a")

        if proba is not None:
            st.metric("ML Probability (+5% target)", f"{proba:.0%}")

        reasoning = s.get("reasoning", [])
        if reasoning:
            st.markdown("**Signals:**")
            for r in reasoning:
                st.markdown(f"- {r}")

        st.markdown(_trade_link(sym, asset_cls))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-asset screener tab renderer
# ─────────────────────────────────────────────────────────────────────────────

def _filt(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """Filter DataFrame to one asset class and drop the asset_class column."""
    if df.empty or "asset_class" not in df.columns:
        return df
    return (
        df[df["asset_class"] == asset_class]
        .drop(columns=["asset_class"])
        .reset_index(drop=True)
    )


def _render_screener_tabs(data: DashboardData, asset_class: str) -> None:
    movers_up     = _filt(data.movers_up, asset_class)
    movers_down   = _filt(data.movers_down, asset_class)
    oversold_df   = _filt(data.oversold, asset_class)
    overbought_df = _filt(data.overbought, asset_class)
    breakouts_df  = _filt(data.breakouts, asset_class)
    all_scores_df = _filt(data.all_scores, asset_class)
    hc = [s for s in data.high_conviction if s.get("asset_class") == asset_class]

    sym_set = data.stock_symbols if asset_class == "stock" else data.crypto_symbols
    prices  = {k: v for k, v in data.prices_dict.items() if k in sym_set}

    tabs = st.tabs([
        "📈 Movers", "🔴 Oversold", "🟠 Overbought",
        "🟢 Breakouts", "🎯 High-Conviction", "📊 All Symbols", "📉 Chart",
    ])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Gainers")
            _show_screener_table(movers_up, "No symbols up ≥5% today.")
        with c2:
            st.subheader("Top Losers")
            _show_screener_table(movers_down, "No symbols down ≥5% today.")

    with tabs[1]:
        st.subheader("Oversold Watch  (RSI < 30)")
        st.caption("Oversold setups historically bounce ~50% of the time. "
                   "Always verify fundamentals before acting.")
        _show_screener_table(oversold_df, "No oversold symbols today.")

    with tabs[2]:
        st.subheader("Overbought  (RSI > 70)")
        st.caption("Potential exit signals if holding; avoid fresh longs.")
        _show_screener_table(overbought_df, "No overbought symbols today.")

    with tabs[3]:
        st.subheader("Volume Breakouts  (≥3× avg)")
        st.caption("Unusual volume often precedes significant moves. "
                   "Does not indicate direction — verify price context.")
        _show_screener_table(breakouts_df, "No volume breakouts today.")

    with tabs[4]:
        st.subheader("High-Conviction Setups  (score ≥ 7/10)")
        if not hc:
            st.info("No high-conviction setups above threshold today.")
        else:
            for s in hc:
                _render_signal_card(s, expanded=True)

    with tabs[5]:
        st.subheader("All Symbols — Composite Scores")
        search = st.text_input(
            "Filter by symbol", placeholder="e.g. AAPL or BTC",
            key=f"search_{asset_class}",
        )
        df_all = all_scores_df.copy()
        if "ml_proba" in df_all.columns:
            df_all["ml_proba"] = df_all["ml_proba"].apply(
                lambda x: x * 100 if x is not None and not (isinstance(x, float) and np.isnan(x)) else None
            )
        if search:
            df_all = df_all[df_all["symbol"].str.contains(search.upper(), case=False, na=False)]
        st.dataframe(
            df_all,
            use_container_width=True,
            hide_index=True,
            column_config=_ALL_SCORES_COL_CONFIG,
        )

    with tabs[6]:
        st.subheader("Price Chart")
        symbols = sorted(prices.keys())
        if not symbols:
            st.info("No price data available.")
        else:
            default_sym = hc[0]["symbol"] if hc else symbols[0]
            default_idx = symbols.index(default_sym) if default_sym in symbols else 0
            selected = st.selectbox("Symbol", symbols, index=default_idx,
                                    key=f"chart_{asset_class}")
            fig = _build_price_chart(selected, prices)
            st.plotly_chart(fig, use_container_width=True)

            df_sym = prices.get(selected)
            if df_sym is not None and not df_sym.empty:
                row  = df_sym.iloc[-1]
                prev = df_sym.iloc[-2] if len(df_sym) > 1 else row
                chg  = (row["close"] - prev["close"]) / prev["close"] * 100
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Close",  f"${row['close']:.2f}")
                sc2.metric("Change", f"{chg:+.1f}%")
                sc3.metric("High",   f"${row['high']:.2f}")
                sc4.metric("Low",    f"${row['low']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Overview tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_overview_tab(data: DashboardData) -> None:
    n_signals = len(data.high_conviction)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Movers Up",   len(data.movers_up),   help="Symbols up ≥5% today")
    m2.metric("Movers Down", len(data.movers_down), help="Symbols down ≥5% today")
    m3.metric("Oversold",    len(data.oversold),    help="RSI < 30 with normal volume")
    m4.metric("Breakouts",   len(data.breakouts),   help="Volume ≥ 3× 20-day average")
    m5.metric("Signals ≥7",  n_signals,             help="Composite score ≥ 7/10")

    st.divider()

    hc_stocks = [s for s in data.high_conviction if s.get("asset_class") == "stock"]
    hc_crypto = [s for s in data.high_conviction if s.get("asset_class") == "crypto"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Stock Signal")
        if hc_stocks:
            _render_signal_card(hc_stocks[0], expanded=True)
        else:
            st.info("No stock signals ≥7 today.")
    with c2:
        st.subheader("Top Crypto Signal")
        if hc_crypto:
            _render_signal_card(hc_crypto[0], expanded=True)
        else:
            st.info("No crypto signals ≥7 today.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 JAI Trading Research")

    if "do_refresh" not in st.session_state:
        st.session_state.do_refresh = False

    data = load_data(refresh=st.session_state.do_refresh)
    if st.session_state.do_refresh:
        st.session_state.do_refresh = False

    st.caption(f"**{data.date}**  ·  {data.universe_size} symbols tracked")

    col_a, col_b, col_c = st.columns([1, 1, 6])
    with col_a:
        if st.button("🔄 Refresh View"):
            st.rerun()
    with col_b:
        if st.button("📥 Pull Fresh Data"):
            st.cache_data.clear()
            st.session_state.do_refresh = True
            st.rerun()

    st.divider()

    top_tabs = st.tabs(["📈 Stocks", "₿ Crypto", "📊 Overview"])

    with top_tabs[0]:
        _render_screener_tabs(data, "stock")

    with top_tabs[1]:
        _render_screener_tabs(data, "crypto")

    with top_tabs[2]:
        _render_overview_tab(data)


if __name__ == "__main__":
    main()
