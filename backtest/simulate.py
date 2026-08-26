from __future__ import annotations
"""
Yeniden kullanılabilir backtest çekirdeği — getiri serisi üretimi, portföy
simülasyonu ve performans metrikleri.

Faz 0 kapsamında analysis_backtest.py'den buraya taşındı (mantık DEĞİŞMEDİ —
sadece modülerleştirildi) ki Faz 1+ script'leri (sinyal doğrulama, ağırlık
kalibrasyonu) aynı simülasyon çekirdeğini tekrar tekrar yazmadan kullanabilsin.

Varsayımlar (basitleştirme, tam finansal mühendislik değil) — analysis_backtest.py
docstring'inde de belgeli:
  mevduat        : o günkü deposit_rate'in günlük tahakkuku (carry — vade riski yok, doğru)
  tahvil         : carry (getiri tahakkuku) + fiyat etkisi (ΔP/P ≈ -durasyon×Δgetiri,
                   durasyon=5 yıl varsayımı). Faz 2'de eklendi — bkz. BOND_DURATION_YEARS
                   yorumu; salt carry, faiz seviyesiyle mekanik korelasyon yaratıyordu.
  doviz          : usdtry günlük % değişimi
  altin          : (gold_ons_usd * usdtry) günlük % değişimi
  hisse          : bist100 günlük % değişimi
  temettu_hisse  : bist100 günlük % değişimi + sabit %4/yıl ek temettü tahakkuku
  yatirim_fonu   : %50 hisse + %50 mevduat karışımı (karma fon yaklaşıklaması)
  kripto         : (btc_usd * usdtry) günlük % değişimi
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import decision_engine as de

DIVIDEND_YIELD_ANNUAL = 0.04   # temettü hisse ek getirisi varsayımı
BOND_DURATION_YEARS = 5.0      # tahvil fiyat-duyarlılığı varsayımı (bkz. aşağıdaki not)


# ─────────────────────────────────────────────────────────────────────────
# 1. Getiri serilerini oluştur
# ─────────────────────────────────────────────────────────────────────────
def build_return_series() -> pd.DataFrame:
    df = pd.read_csv(de.MERGED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    r = pd.DataFrame(index=df.index)
    r["mevduat"] = (1 + df["deposit_rate"] / 100) ** (1 / 365) - 1

    # FAZ 2 düzeltmesi: tahvil eskiden SADECE carry (getiri oranı tahakkuku) kullanıyordu.
    # Bu, faiz seviyesiyle mekanik bir korelasyon yaratıyor (yüksek getiri = yüksek ölçülen
    # "getiri", tanım gereği) — Faz 1'deki real_rate/us10y->tahvil'in aşırı yüksek IC'sinin
    # (0.70+) büyük kısmı muhtemelen bu artifakttan kaynaklanıyordu, gerçek öngörü gücünden
    # değil. Artık fiyat etkisi de ekleniyor: ΔP/P ≈ -durasyon × Δgetiri (standart yaklaşım).
    bond_yield = df.get("turkey_10y_bond", df["deposit_rate"])
    carry_daily = (1 + bond_yield / 100) ** (1 / 365) - 1
    yield_chg_daily = bond_yield.diff() / 100   # puan -> ondalık
    price_return_daily = -BOND_DURATION_YEARS * yield_chg_daily
    r["tahvil"] = carry_daily + price_return_daily

    r["doviz"]   = df["usdtry"].pct_change(fill_method=None)
    r["altin"]   = (df["gold_ons_usd"] * df["usdtry"]).pct_change(fill_method=None)
    r["hisse"]   = df["bist100"].pct_change(fill_method=None)
    r["temettu_hisse"] = r["hisse"] + ((1 + DIVIDEND_YIELD_ANNUAL) ** (1 / 365) - 1)
    r["yatirim_fonu"]  = 0.5 * r["hisse"] + 0.5 * r["mevduat"]
    r["kripto"]  = (df["btc_usd"] * df["usdtry"]).pct_change(fill_method=None)

    return r[de.ASSETS].dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────
# 2. Simülasyon: rebalans tarihlerinde ağırlık değiştir, günlük getiri bileşimi
# ─────────────────────────────────────────────────────────────────────────
def simulate(returns: pd.DataFrame, rebalance_dates: list[pd.Timestamp],
             weights_at: dict[pd.Timestamp, dict]) -> pd.Series:
    """weights_at: {rebalance_date: {asset: pct}} — her rebalans tarihinde kullanılacak ağırlık."""
    value = 1.0
    curve = {}
    dates = returns.index
    sorted_rb = sorted(weights_at.keys())

    current_w = None
    rb_idx = 0
    for d in dates:
        while rb_idx < len(sorted_rb) and sorted_rb[rb_idx] <= d:
            current_w = {a: weights_at[sorted_rb[rb_idx]].get(a, 0) / 100 for a in de.ASSETS}
            rb_idx += 1
        if current_w is None:
            curve[d] = value
            continue
        day_ret = sum(current_w[a] * (returns.loc[d, a] if pd.notna(returns.loc[d, a]) else 0)
                      for a in de.ASSETS)
        value *= (1 + day_ret)
        curve[d] = value
    return pd.Series(curve)


# ─────────────────────────────────────────────────────────────────────────
# 3. Metrikler
# ─────────────────────────────────────────────────────────────────────────
def compute_metrics(curve: pd.Series, risk_free_annual: float) -> dict:
    daily_ret = curve.pct_change().dropna()
    n_days = len(curve)
    total_return = curve.iloc[-1] / curve.iloc[0] - 1
    years = n_days / 252
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    sharpe = (daily_ret.mean() - rf_daily) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else np.nan
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    sortino = ((daily_ret.mean() - rf_daily) / downside.std() * np.sqrt(252)
               if len(downside) > 1 and downside.std() > 0 else np.nan)
    var_95 = daily_ret.quantile(0.05) * 100
    cvar_95 = daily_ret[daily_ret <= daily_ret.quantile(0.05)].mean() * 100
    return {
        "total_return_%": total_return * 100,
        "cagr_%": cagr * 100,
        "ann_vol_%": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_%": max_dd * 100,
        "calmar": calmar,
        "var_95_%": var_95,
        "cvar_95_%": cvar_95,
    }
