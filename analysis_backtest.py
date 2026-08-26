from __future__ import annotations
"""
Karar motorunun performans analizi — "tahminler mantıklı mı?" sorusuna somut
metriklerle cevap.

Simülasyon çekirdeği backtest/ paketine taşındı (bkz. backtest/simulate.py,
backtest/split.py) — bu script artık sadece tam-dönem (2020-2026) DİNAMİK vs
STATİK vs %100 MEVDUAT karşılaştırmasını çalıştıran bir CLI. Walk-forward
(train/test ayrımlı) analizler için backtest.split kullanan Faz 1/2
script'lerine bakın.

  DİNAMİK  : her ay decision_engine.run_engine() çağrılıp final_alloc kullanılır
  STATİK   : her ay sabit BASE_PORTFOLIOS[profile] kullanılır (motor hiç tilt yapmaz)

Karşılaştırma STATİK'i taban alır — asıl soru "skor bazlı dinamik ayarlama,
sabit taban portföyden DAHA Mİ İYİ" (risk-ayarlı getiri anlamında).

Varsayımlar için bkz. backtest/simulate.py docstring.

Çalıştır: python analysis_backtest.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import decision_engine as de
from backtest import build_return_series, simulate, compute_metrics


def run_analysis():
    print("=" * 78)
    print("  KARAR MOTORU PERFORMANS ANALİZİ — Dinamik vs Statik Backtest")
    print("=" * 78)

    returns = build_return_series()
    avg_deposit_rate = pd.read_csv(de.MERGED_CSV, parse_dates=["date"])["deposit_rate"].mean() / 100

    all_dates = de.get_available_dates()
    all_dates_ts = pd.to_datetime(all_dates)
    month_ends = pd.Series(all_dates_ts).groupby(pd.Series(all_dates_ts).dt.to_period("M")).max().tolist()
    # ilk 260 gün (yaklaşık 1 yıl) rolling window ısınması için atla
    month_ends = [d for d in month_ends if d >= all_dates_ts.min() + pd.Timedelta(days=260)]

    print(f"\nVeri aralığı : {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Rebalans     : {len(month_ends)} ay-sonu tarihi, aylık")
    print(f"Risksiz oran : ortalama mevduat faizi = %{avg_deposit_rate*100:.1f} (Sharpe için)")
    print("               NOT: TR mevduat faizi bu dönemde çok geniş bir bantta gezdi —")
    print("               Sharpe'ların hepsi negatif çıkarsa şaşırmayın, bu enflasyonla")
    print("               mücadele döneminde nakidin nominal getirisi zaten çok yüksekti.")

    results = {}
    for profile in ["az_riskli", "orta_riskli", "cok_riskli"]:
        print(f"\n{'-'*78}\n  PROFİL: {profile.upper()}\n{'-'*78}")

        dyn_weights, stat_weights = {}, {}
        risk_scores_by_date = {}
        for d in month_ends:
            r = de.run_engine(profile, date=d.strftime("%Y-%m-%d"))
            dyn_weights[d] = r["final_alloc"]
            stat_weights[d] = de.BASE_PORTFOLIOS[profile]
            risk_scores_by_date[d] = r["risk_score"]

        mevduat_only = {d: {"mevduat": 100} for d in month_ends}
        dyn_curve = simulate(returns, month_ends, dyn_weights)
        stat_curve = simulate(returns, month_ends, stat_weights)
        cash_curve = simulate(returns, month_ends, mevduat_only)

        dyn_m = compute_metrics(dyn_curve, avg_deposit_rate)
        stat_m = compute_metrics(stat_curve, avg_deposit_rate)
        cash_m = compute_metrics(cash_curve, avg_deposit_rate)

        print(f"{'Metrik':<20}{'Dinamik':>14}{'Statik (taban)':>18}{'Fark':>12}{'%100 Mevduat':>16}")
        for k in ["total_return_%", "cagr_%", "ann_vol_%", "sharpe", "sortino",
                  "max_drawdown_%", "calmar", "var_95_%", "cvar_95_%"]:
            dv, sv, cv = dyn_m[k], stat_m[k], cash_m[k]
            diff = dv - sv
            print(f"{k:<20}{dv:>14.2f}{sv:>18.2f}{diff:>+12.2f}{cv:>16.2f}")

        # Aylık kazan/kaybet oranı (dinamik o ay statikten iyi mi?)
        dyn_monthly = dyn_curve.resample("ME").last().pct_change().dropna()
        stat_monthly = stat_curve.resample("ME").last().pct_change().dropna()
        common = dyn_monthly.index.intersection(stat_monthly.index)
        win_rate = (dyn_monthly[common] > stat_monthly[common]).mean() * 100
        print(f"\nDinamik'in statik'i geçtiği ay oranı: %{win_rate:.1f} ({len(common)} ay)")

        results[profile] = {"dyn": dyn_m, "stat": stat_m, "win_rate": win_rate,
                            "risk_scores": risk_scores_by_date}

    # ── Risk skoru öngörü gücü testi ──────────────────────────────────────
    print(f"\n{'='*78}\n  RİSK SKORU ÖNGÖRÜ GÜCÜ TESTİ\n{'='*78}")
    print("Soru: risk_score(t) yüksekken, piyasa ileriki 20 işgününde gerçekten")
    print("zayıf mı performans gösteriyor? (beklenen: NEGATİF korelasyon)\n")

    bist = returns["hisse"].dropna()
    fwd_20d = (1 + bist).rolling(20).apply(np.prod, raw=True).shift(-20) - 1

    rs = results["orta_riskli"]["risk_scores"]
    rs_series = pd.Series(rs).sort_index()
    aligned = pd.DataFrame({"risk_score": rs_series}).reindex(fwd_20d.index, method="ffill")
    aligned["fwd_20d_bist"] = fwd_20d
    aligned = aligned.dropna()
    corr = aligned["risk_score"].corr(aligned["fwd_20d_bist"])
    print(f"korelasyon(risk_score, ileri_20g_BIST_getirisi) = {corr:+.3f}")
    print(f"({'öngörücü sinyal - beklenen yönde' if corr < -0.05 else 'zayıf/sıfır ilişki - risk_score forward BIST getirisini açıklamıyor' if abs(corr) <= 0.05 else 'BEKLENENİN TERSİ yönde (şüpheli)'})")

    # ── Genel değerlendirme ──────────────────────────────────────────────
    print(f"\n{'='*78}\n  GENEL DEĞERLENDİRME\n{'='*78}")
    n_better = sum(1 for p in results if results[p]["dyn"]["sharpe"] > results[p]["stat"]["sharpe"])
    print(f"Dinamik motorun Sharpe'ı statik tabandan iyi olduğu profil sayısı: {n_better}/3")
    for p in results:
        d, s = results[p]["dyn"], results[p]["stat"]
        verdict = "EKLİYOR" if d["sharpe"] > s["sharpe"] else "EKLEMİYOR / negatif"
        print(f"  {p:<14}: Sharpe {d['sharpe']:.2f} vs {s['sharpe']:.2f}  -> dinamik ayar değer {verdict}")

    return results


if __name__ == "__main__":
    run_analysis()
