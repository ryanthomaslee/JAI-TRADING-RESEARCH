"""
Daily market intelligence report generator.

Produces output in two formats simultaneously:
  1. Terminal (ANSI-clean, truncated to REPORT_TERMINAL_MAX_ROWS per section)
  2. Markdown file (full output, all entries)

Design decisions:
  Scannable in 60 seconds: sections are short, symbols are one-liners.
  The markdown file has the full data for deep-dives.

  Honest language: "historically bounces ~50% of the time" not
  "this WILL bounce."  The screener identifies conditions, not outcomes.
"""

from __future__ import annotations

import textwrap
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import REPORTS_DIR, REPORT_TERMINAL_MAX_ROWS


# ─────────────────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " n/a"
    return f"{v:+.1f}%"


def _price(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    n/a"
    return f"${v:>8.2f}"


def _rsi_label(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "RSI  n/a"
    return f"RSI {v:.0f}"


def _vol_label(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "vol  n/a"
    return f"vol {v:.1f}x"


def _one_liner(row) -> str:
    sym   = str(row["symbol"])[:12].ljust(12)
    price = _price(row.get("current_price"))
    chg   = _pct(row.get("change_pct"))
    vol   = _vol_label(row.get("volume_ratio"))
    rsi   = _rsi_label(row.get("rsi"))
    return f"  {sym}  {price}  {chg:>7}  {vol:>10}  {rsi}"


def _truncate_note(n_total: int, n_shown: int, section: str) -> str:
    extra = n_total - n_shown
    if extra <= 0:
        return ""
    return f"  … +{extra} more in {section} of the markdown report."


# ─────────────────────────────────────────────────────────────────────────────
#  Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_movers(
    movers_up:   pd.DataFrame,
    movers_down: pd.DataFrame,
    max_each:    int = 5,
) -> tuple[str, str]:
    """Returns (terminal_str, markdown_str) for today's movers section."""
    lines_t = ["## ⚡ TODAY'S MOVERS\n"]
    lines_m = ["## ⚡ TODAY's Movers\n"]

    header = f"  {'Symbol':<12}  {'Price':>9}  {'Change':>7}  {'Volume':>10}  {'RSI'}"

    # Up movers
    top_up   = movers_up.head(max_each)
    total_up = len(movers_up)
    if not top_up.empty:
        lines_t.append("  ### Top gainers")
        lines_m.append("### Top gainers")
        lines_t.append(header)
        lines_m.append(header)
        for _, row in top_up.iterrows():
            lines_t.append(_one_liner(row))
            lines_m.append(_one_liner(row))
        if total_up > max_each:
            note = _truncate_note(total_up, max_each, "Movers")
            lines_t.append(note)
    else:
        lines_t.append("  No symbols up ≥5% today.")

    lines_t.append("")
    lines_m.append("")

    # Down movers
    top_down   = movers_down.head(max_each)
    total_down = len(movers_down)
    if not top_down.empty:
        lines_t.append("  ### Top losers")
        lines_m.append("### Top losers")
        lines_t.append(header)
        lines_m.append(header)
        for _, row in top_down.iterrows():
            lines_t.append(_one_liner(row))
            lines_m.append(_one_liner(row))
        if total_down > max_each:
            note = _truncate_note(total_down, max_each, "Movers")
            lines_t.append(note)
    else:
        lines_t.append("  No symbols down ≥5% today.")

    return "\n".join(lines_t), "\n".join(lines_m)


def _section_oversold(
    oversold_df: pd.DataFrame,
    max_rows:    int = REPORT_TERMINAL_MAX_ROWS,
) -> tuple[str, str]:
    lines_t = [
        "## 🔴 OVERSOLD WATCH  (RSI < 30, volume ≥ 0.8× avg)\n",
        "  Oversold setups historically bounce ~50% of the time.",
        "  Always verify fundamentals before acting.\n",
    ]
    lines_m = [
        "## 🔴 Oversold Watch (RSI < 30)\n",
        "> Oversold setups historically bounce ~50% of the time. "
        "This is a setup flag, not a buy signal.\n",
    ]

    if oversold_df.empty:
        lines_t.append("  No oversold setups today.")
        lines_m.append("No oversold setups today.")
        return "\n".join(lines_t), "\n".join(lines_m)

    total  = len(oversold_df)
    shown  = oversold_df.head(max_rows)
    header = f"  {'Symbol':<12}  {'Price':>9}  {'Change':>7}  {'Volume':>10}  {'RSI'}"
    lines_t.append(header)
    lines_m.append(header)
    for _, row in shown.iterrows():
        lines_t.append(_one_liner(row))
        lines_m.append(_one_liner(row))
    if total > max_rows:
        lines_t.append(_truncate_note(total, max_rows, "Oversold Watch"))

    return "\n".join(lines_t), "\n".join(lines_m)


def _section_breakouts(
    breakouts_df: pd.DataFrame,
    max_rows:     int = REPORT_TERMINAL_MAX_ROWS,
) -> tuple[str, str]:
    lines_t = [
        "## 🟢 VOLUME BREAKOUTS  (volume ≥ 3× avg)\n",
        "  Unusual volume often precedes significant moves.",
        "  Does NOT indicate direction — verify price context.\n",
    ]
    lines_m = [
        "## 🟢 Volume Breakouts (≥3× avg)\n",
        "> High volume does not indicate direction. "
        "Always combine with price trend context.\n",
    ]

    if breakouts_df.empty:
        lines_t.append("  No volume breakouts today.")
        lines_m.append("No volume breakouts today.")
        return "\n".join(lines_t), "\n".join(lines_m)

    total  = len(breakouts_df)
    shown  = breakouts_df.head(max_rows)
    header = f"  {'Symbol':<12}  {'Price':>9}  {'Change':>7}  {'Volume':>10}  {'RSI'}"
    lines_t.append(header)
    lines_m.append(header)
    for _, row in shown.iterrows():
        lines_t.append(_one_liner(row))
        lines_m.append(_one_liner(row))
    if total > max_rows:
        lines_t.append(_truncate_note(total, max_rows, "Volume Breakouts"))

    return "\n".join(lines_t), "\n".join(lines_m)


def _section_new_listings(new_df: pd.DataFrame) -> tuple[str, str]:
    lines_t = [
        "## 🆕 NEW / THIN LISTINGS  (< 90 days of history)\n",
        "  Limited data — analytics are unreliable.",
        "  RSI/SMA indicators need 200+ bars to stabilise.\n",
    ]
    lines_m = [
        "## 🆕 New / Thin Listings\n",
        "> **⚠ Limited data — high risk. Analytics unreliable below 200 bars.**\n",
    ]

    if new_df.empty:
        lines_t.append("  All universe symbols have sufficient history.")
        lines_m.append("All universe symbols have sufficient history.")
        return "\n".join(lines_t), "\n".join(lines_m)

    for _, row in new_df.iterrows():
        sym  = str(row["symbol"])
        days = int(row["change_pct"]) if not np.isnan(row["change_pct"]) else "?"
        p    = _price(row.get("current_price"))
        line = f"  {sym:<14}  {p}  ({days} days of history)  ⚠ limited data"
        lines_t.append(line)
        lines_m.append(line)

    return "\n".join(lines_t), "\n".join(lines_m)


def _section_overbought(
    overbought_df: pd.DataFrame,
    max_rows:      int = 10,
) -> tuple[str, str]:
    lines_t = [
        "## 🟡 OVERBOUGHT  (RSI > 70)\n",
        "  Potential exit signals if holding; avoid fresh longs.\n",
    ]
    lines_m = [
        "## 🟡 Overbought (RSI > 70)\n",
        "> Potential exit signals if holding; avoid fresh longs.\n",
    ]

    if overbought_df.empty:
        lines_t.append("  No overbought symbols today.")
        lines_m.append("No overbought symbols today.")
        return "\n".join(lines_t), "\n".join(lines_m)

    total  = len(overbought_df)
    shown  = overbought_df.head(max_rows)
    header = f"  {'Symbol':<12}  {'Price':>9}  {'Change':>7}  {'Volume':>10}  {'RSI'}"
    lines_t.append(header)
    lines_m.append(header)
    for _, row in shown.iterrows():
        lines_t.append(_one_liner(row))
        lines_m.append(_one_liner(row))
    if total > max_rows:
        extra = total - max_rows
        note  = f"  … +{extra} more overbought symbols in the markdown report."
        lines_t.append(note)
        lines_m.append(f"  … +{extra} more symbols not shown.")

    return "\n".join(lines_t), "\n".join(lines_m)


def _section_high_conviction(scores: list[dict], threshold: float = 7.0) -> tuple[str, str]:
    """
    High-conviction setups: symbols with composite score >= threshold.
    Renamed from ML SIGNALS since most symbols use technical-only scoring.
    """
    lines_t = [f"## 🎯 HIGH-CONVICTION SETUPS  (score ≥ {threshold:.0f}/10)\n"]
    lines_m = [f"## 🎯 High-Conviction Setups (score ≥ {threshold:.0f}/10)\n"]

    high_conviction = [s for s in scores if s.get("score", 0) >= threshold]

    if not high_conviction:
        msg = "  No high-conviction setups above threshold today."
        lines_t.append(msg)
        lines_m.append(msg.strip())
        return "\n".join(lines_t), "\n".join(lines_m)

    for s in high_conviction:
        sym      = s["symbol"]
        score    = s.get("score", 0)
        proba    = s.get("ml_proba")
        rsi      = s.get("rsi", float("nan"))
        vol      = s.get("volume_ratio", float("nan"))
        has_ml   = s.get("has_ml", False)
        label    = "(ML + technical)" if has_ml else "(technical-only)"

        lines_t.append(f"  ── {sym}  Score {score:.1f}/10  {label}")
        lines_m.append(f"### {sym}  —  Score {score:.1f}/10  {label}")

        if proba is not None:
            lines_t.append(f"     ML probability of +5% target: {proba:.1%}")
            lines_m.append(f"- ML probability of +5% target: **{proba:.1%}**")

        lines_t.append(f"     RSI: {rsi:.0f}  |  Volume: {_vol_label(vol)}")
        lines_m.append(f"- RSI: {rsi:.0f}  |  Volume: {_vol_label(vol)}")

        for reason in s.get("reasoning", []):
            lines_t.append(f"     • {reason}")
            lines_m.append(f"  - {reason}")

        lines_t.append("")
        lines_m.append("")

    return "\n".join(lines_t), "\n".join(lines_m)


# Keep old name as alias so existing tests don't break
_section_ml_signals = _section_high_conviction


def _section_summary(
    n_universe:   int,
    n_up:         int,
    n_down:       int,
    n_oversold:   int,
    n_overbought: int,
    n_breakouts:  int,
    n_new:        int,
    n_ml_signals: int,
    report_date:  str,
) -> tuple[str, str]:
    lines = [
        "## 📊 SUMMARY\n",
        f"  Universe: {n_universe} symbols  |  As of: {report_date}",
        f"  Movers: {n_up} up ≥5%, {n_down} down ≥5%",
        f"  RSI: {n_oversold} oversold (<30), {n_overbought} overbought (>70)",
        f"  Volume breakouts: {n_breakouts}",
        f"  New/thin listings: {n_new}",
        f"  High-conviction setups (≥7): {n_ml_signals}",
    ]
    return "\n".join(lines), "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entrypoints
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    report_date:  _date | str,
    prices_dict:  dict[str, pd.DataFrame],
    movers_up:    pd.DataFrame,
    movers_down:  pd.DataFrame,
    oversold_df:  pd.DataFrame,
    overbought_df: pd.DataFrame,
    breakouts_df: pd.DataFrame,
    new_df:       pd.DataFrame,
    scores:       list[dict],
    ml_threshold: float = 7.0,
) -> tuple[str, str]:
    """
    Build terminal and markdown report strings.

    Returns (terminal_str, markdown_str).
    """
    date_str = str(report_date)
    n_uni    = len(prices_dict)

    title_t  = f"\n{'═'*64}\n  MARKET INTELLIGENCE — {date_str}  ({n_uni} symbols)\n{'═'*64}\n"
    title_m  = f"# Market Intelligence — {date_str}\n\n> Universe: {n_uni} symbols\n"

    s_movers_t,    s_movers_m    = _section_movers(movers_up, movers_down)
    s_oversold_t,  s_oversold_m  = _section_oversold(oversold_df)
    s_overbought_t, s_overbought_m = _section_overbought(overbought_df)
    s_breaks_t,    s_breaks_m    = _section_breakouts(breakouts_df)
    s_new_t,       s_new_m       = _section_new_listings(new_df)
    s_hc_t,        s_hc_m        = _section_high_conviction(scores, threshold=ml_threshold)

    n_hc_sig = sum(1 for s in scores if s.get("score", 0) >= ml_threshold)
    s_sum_t, s_sum_m = _section_summary(
        n_universe   = n_uni,
        n_up         = len(movers_up),
        n_down       = len(movers_down),
        n_oversold   = len(oversold_df),
        n_overbought = len(overbought_df),
        n_breakouts  = len(breakouts_df),
        n_new        = len(new_df),
        n_ml_signals = n_hc_sig,
        report_date  = date_str,
    )

    sep = "\n" + "─"*64 + "\n"
    terminal  = title_t  + sep.join([s_movers_t, s_oversold_t, s_overbought_t,
                                      s_breaks_t, s_new_t, s_hc_t, s_sum_t])
    markdown  = title_m  + "\n---\n".join([s_movers_m, s_oversold_m, s_overbought_m,
                                           s_breaks_m, s_new_m, s_hc_m, s_sum_m])
    return terminal, markdown


def save_report(markdown: str, report_date: _date | str) -> Path:
    """Save markdown report to data/reports/YYYY-MM-DD.md. Returns path."""
    path = REPORTS_DIR / f"{report_date}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
