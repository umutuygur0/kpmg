from __future__ import annotations
"""
FAZ 1 — Sinyal Doğrulama: Information Coefficient (IC) testi.

Soru: decision_engine.py'nin skor formüllerinde kullanılan her ham sinyal,
GERÇEKTEN ilgili varlığın ileriki getirisini açıklıyor mu, yoksa gürültü mü?

Yöntem: her (sinyal, hedef_varlık) çifti için sinyal(t) ile hedef_varlığın
ileri N-günlük getirisi arasında Spearman korelasyonu (IC) + p-değeri
hesaplanır. Spearman, sıralama bazlı olduğu için ham değer ile onun
percentile-rank dönüşümü (motorun asıl kullandığı rr_pct/fxs_pct/... gibi)
arasında -aynı pencere kullanıldığı sürece- fark yaratmaz; bu yüzden
doğrudan ham sinyal üzerinden test ediliyor (basitleştirme, belgelenmiştir).

Karar kuralı: p > 0.10 ise sinyal "gürültüden ayrılamıyor" sayılır (AT).
Bu eşik Faz 2'nin ağırlık öğrenmesine hangi sinyallerin gireceğini belirler.

Sinyal → hedef varlık eşlemesi, compute_asset_scores()'daki GERÇEK formüllerden
çıkarıldı (decision_engine.py):
  real_rate        -> mevduat, tahvil, yatirim_fonu, temettu_hisse   (rr_pct kullanılıyor)
  fx_stress        -> doviz, altin                                   (fxs_pct)
  bist_momentum    -> yatirim_fonu, hisse, temettu_hisse              (bist_pct)
  gold_real_return -> altin                                          (grr_pct)
  us10y            -> tahvil, kripto                                 (us10_inv)
  risk_score       -> TÜM varlıklar (her formülde risk terimi var)
  vix_level        -> hisse (risk_score'un bileşeni, doğrudan da test edilir)
  cds_level        -> hisse, doviz (risk_score'un bileşeni)

Çalıştır: python signal_validation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import decision_engine as de
from backtest import build_return_series

FORWARD_HORIZONS = [5, 20, 60]   # işgünü
P_VALUE_THRESHOLD = 0.10

SIGNAL_ASSET_MAP = {
    "real_rate":        ["mevduat", "tahvil", "yatirim_fonu", "temettu_hisse"],
    "fx_stress":        ["doviz", "altin"],
    "bist_momentum":    ["yatirim_fonu", "hisse", "temettu_hisse"],
    "gold_real_return": ["altin"],
    "us10y":            ["tahvil", "kripto"],
    "risk_score":       de.ASSETS,
    "vix_level":        ["hisse"],
    "cds_level":        ["hisse", "doviz"],
}


def build_signal_frame() -> pd.DataFrame:
    df = pd.read_csv(de.MERGED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d = de.compute_derived(df)
    d = d.set_index("date")
    return d


def forward_return(returns: pd.Series, horizon: int) -> pd.Series:
    """t anındaki t+1..t+horizon bileşik getirisi (leakage yok — t'nin kendisi dahil değil)."""
    fwd = (1 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    return fwd


def run_validation() -> pd.DataFrame:
    print("=" * 90)
    print("  FAZ 1 — SİNYAL DOĞRULAMA (Information Coefficient)")
    print("=" * 90)

    signals = build_signal_frame()
    returns = build_return_series()

    rows = []
    for signal_name, assets in SIGNAL_ASSET_MAP.items():
        if signal_name not in signals.columns:
            continue
        sig = signals[signal_name]
        for asset in assets:
            if asset not in returns.columns:
                continue
            for horizon in FORWARD_HORIZONS:
                fwd = forward_return(returns[asset], horizon)
                aligned = pd.DataFrame({"signal": sig, "fwd": fwd}).dropna()
                if len(aligned) < 60:
                    continue
                ic, pval = spearmanr(aligned["signal"], aligned["fwd"])
                rows.append({
                    "signal": signal_name, "asset": asset, "horizon_gun": horizon,
                    "ic": ic, "p_value": pval, "n": len(aligned),
                    "karar": "TUT" if pval <= P_VALUE_THRESHOLD else "AT",
                })

    result = pd.DataFrame(rows).sort_values(["signal", "asset", "horizon_gun"]).reset_index(drop=True)

    print(f"\n{'Sinyal':<18}{'Varlık':<16}{'Ufuk(g)':>8}{'IC':>9}{'p-değeri':>11}{'n':>7}{'Karar':>8}")
    print("-" * 90)
    for _, r in result.iterrows():
        flag = "  <-- ŞÜPHELİ (ters yönlü, anlamlı)" if r["karar"] == "TUT" and r["ic"] > 0 and r["signal"] in (
            "risk_score", "vix_level", "cds_level", "fx_stress") and r["asset"] in ("hisse", "kripto", "temettu_hisse") else ""
        print(f"{r['signal']:<18}{r['asset']:<16}{r['horizon_gun']:>8}{r['ic']:>9.3f}{r['p_value']:>11.4f}{r['n']:>7}{r['karar']:>8}{flag}")

    n_tut = (result["karar"] == "TUT").sum()
    n_at = (result["karar"] == "AT").sum()
    print(f"\nToplam test: {len(result)}  |  TUT (anlamlı): {n_tut}  |  AT (gürültü): {n_at}")

    print(f"\n{'='*90}\n  SİNYAL BAZINDA ÖZET (en iyi ufuktaki |IC|)\n{'='*90}")
    summary = (result.loc[result.groupby("signal")["ic"].apply(lambda s: s.abs().idxmax())]
               .sort_values("ic", key=lambda s: s.abs(), ascending=False))
    print(f"{'Sinyal':<18}{'En güçlü varlık':<16}{'Ufuk':>6}{'IC':>9}{'p-değeri':>11}{'Genel karar':>14}")
    for _, r in summary.iterrows():
        sig_rows = result[result["signal"] == r["signal"]]
        overall = "EN AZ 1 ANLAMLI" if (sig_rows["karar"] == "TUT").any() else "TÜM TESTLER GÜRÜLTÜ"
        print(f"{r['signal']:<18}{r['asset']:<16}{r['horizon_gun']:>6}{r['ic']:>9.3f}{r['p_value']:>11.4f}{overall:>14}")

    return result


if __name__ == "__main__":
    run_validation()
