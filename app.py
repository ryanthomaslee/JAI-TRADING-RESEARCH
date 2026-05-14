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
#  Table formatter helpers
# ─────────────────────────────────────────────────────────────────────────────

_SCREENER_COL_CONFIG = {
    "symbol":        st.column_config.TextColumn("Symbol", width="small"),
    "current_price": st.column_config.NumberColumn("Price", format="$%.2f", width="small"),
    "change_pct":    st.column_config.NumberColumn("Change %", format="%.1f%%", width="small"),
    "volume_ratio":  st.column_config.NumberColumn("Vol Ratio", format="%.1fx", width="small"),
    "rsi":           st.column_config.NumberColumn("RSI", format="%.0f", width="small"),
    "category_score":st.column_config.NumberColumn("Score", format="%.0f", width="small"),
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


def _show_screener_table(df: pd.DataFrame, empty_msg: str) -> None:
    if df.empty:
        st.info(empty_msg)
        return
    display = df.copy()
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


# ─────────────────────────────────────────────────────────────────────────────
#  Chart builder
# ─────────────────────────────────────────────────────────────────────────────

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

    # Price line
    fig.add_trace(go.Scatter(
        x=dates, y=tail["close"],
        name="Close",
        line=dict(color="#4C9BE8", width=2),
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # SMA 20
    fig.add_trace(go.Scatter(
        x=dates, y=tail["sma20"],
        name="SMA 20",
        line=dict(color="#F0A500", width=1.2, dash="dot"),
        hovertemplate="SMA20: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # SMA 50
    fig.add_trace(go.Scatter(
        x=dates, y=tail["sma50"],
        name="SMA 50",
        line=dict(color="#9B59B6", width=1.2, dash="dot"),
        hovertemplate="SMA50: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Volume bars — color by price direction
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

    # Today marker
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
#  Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("📊 JAI Trading Research")

    # Initialise session state for refresh trigger
    if "do_refresh" not in st.session_state:
        st.session_state.do_refresh = False

    data = load_data(refresh=st.session_state.do_refresh)
    # Reset after consuming
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

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_signals = len(data.high_conviction)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Movers Up",    len(data.movers_up),   help="Symbols up ≥5% today")
    m2.metric("Movers Down",  len(data.movers_down), help="Symbols down ≥5% today")
    m3.metric("Oversold",     len(data.oversold),    help="RSI < 30 with normal volume")
    m4.metric("Breakouts",    len(data.breakouts),   help="Volume ≥ 3× 20-day average")
    m5.metric("Signals ≥7",   n_signals,             help="Composite score ≥ 7/10")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Movers",
        "🔴 Oversold",
        "🟠 Overbought",
        "🟢 Breakouts",
        "🎯 High-Conviction",
        "📊 All Symbols",
        "📉 Chart",
    ])

    # ── Tab: Movers ───────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Gainers")
            _show_screener_table(data.movers_up, "No symbols up ≥5% today.")
        with c2:
            st.subheader("Top Losers")
            _show_screener_table(data.movers_down, "No symbols down ≥5% today.")

    # ── Tab: Oversold ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Oversold Watch  (RSI < 30)")
        st.caption("Oversold setups historically bounce ~50% of the time. "
                   "Always verify fundamentals before acting.")
        _show_screener_table(data.oversold, "No oversold symbols today.")

    # ── Tab: Overbought ───────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Overbought  (RSI > 70)")
        st.caption("Potential exit signals if holding; avoid fresh longs.")
        _show_screener_table(data.overbought, "No overbought symbols today.")

    # ── Tab: Breakouts ────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Volume Breakouts  (≥3× avg)")
        st.caption("Unusual volume often precedes significant moves. "
                   "Does not indicate direction — verify price context.")
        _show_screener_table(data.breakouts, "No volume breakouts today.")

    # ── Tab: High-Conviction ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("High-Conviction Setups  (score ≥ 7/10)")
        if not data.high_conviction:
            st.info("No high-conviction setups above threshold today.")
        else:
            for s in data.high_conviction:
                sym    = s["symbol"]
                score  = s.get("score", 0)
                has_ml = s.get("has_ml", False)
                proba  = s.get("ml_proba")
                rsi    = s.get("rsi", float("nan"))
                vol    = s.get("volume_ratio", float("nan"))
                label  = "ML + technical" if has_ml else "technical-only"

                with st.expander(f"**{sym}**  —  Score {score:.1f}/10  ({label})", expanded=True):
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

    # ── Tab: All Symbols ──────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("All Symbols — Composite Scores")

        search = st.text_input("Filter by symbol", placeholder="e.g. AAPL or BTC")
        df_all = data.all_scores.copy()

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

    # ── Tab: Chart ────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Price Chart")
        symbols = sorted(data.prices_dict.keys())

        # Default to the top high-conviction symbol, else first symbol
        default_sym = data.high_conviction[0]["symbol"] if data.high_conviction else symbols[0]
        default_idx = symbols.index(default_sym) if default_sym in symbols else 0

        selected = st.selectbox("Symbol", symbols, index=default_idx)
        fig = _build_price_chart(selected, data.prices_dict)
        st.plotly_chart(fig, use_container_width=True)

        # Show last row of data as a quick stats strip
        df_sym = data.prices_dict.get(selected)
        if df_sym is not None and not df_sym.empty:
            row = df_sym.iloc[-1]
            prev = df_sym.iloc[-2] if len(df_sym) > 1 else row
            chg  = (row["close"] - prev["close"]) / prev["close"] * 100
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Close",  f"${row['close']:.2f}")
            sc2.metric("Change", f"{chg:+.1f}%")
            sc3.metric("High",   f"${row['high']:.2f}")
            sc4.metric("Low",    f"${row['low']:.2f}")


if __name__ == "__main__":
    main()
