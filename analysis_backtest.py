from __future__ import annotations
"""
Karar motorunun performans analizi — "tahminler mantıklı mı?" sorusuna somut
metriklerle cevap.

Yöntem: her varlık sınıfı için gerçek piyasa verisinden (portfolio_data/) günlük
TL-cinsi getiri serisi türetilir (bkz. ASSET_RETURN_ASSUMPTIONS), sonra iki
strateji aylık olarak (ayın son iş günü) yeniden dengelenerek simüle edilir:

  DİNAMİK  : her ay decision_engine.run_engine() çağrılıp final_alloc kullanılır
  STATİK   : her ay sabit BASE_PORTFOLIOS[profile] kullanılır (motor hiç tilt yapmaz)

Karşılaştırma STATİK'i taban alır — asıl soru "skor bazlı dinamik ayarlama,
sabit taban portföyden DAHA Mİ İYİ" (risk-ayarlı getiri anlamında).

Ayrıca motorun risk_score'unun gerçekten öngörücü olup olmadığı ayrı bir testle
kontrol edilir: risk_score(t) ile ileri 20 iş günü BIST getirisi arasındaki
korelasyon (beklenti: negatif — risk yüksekse piyasa zayıf performans göstermeli).

ÖNEMLİ — Varsayımlar (basitleştirme, tam finansal mühendislik değil):
  mevduat        : o günkü deposit_rate'in günlük tahakkuku
  tahvil         : o günkü turkey_10y_bond getirisinin günlük tahakkuku (fiyat/
                   durasyon etkisi yok sayıldı — sadece carry)
  doviz          : usdtry günlük % değişimi
  altin          : (gold_ons_usd * usdtry) günlük % değişimi
  hisse          : bist100 günlük % değişimi
  temettu_hisse  : bist100 günlük % değişimi + sabit %4/yıl ek temettü tahakkuku
  yatirim_fonu   : %50 hisse + %50 mevduat karışımı (karma fon yaklaşıklaması)
  kripto         : (btc_usd * usdtry) günlük % değişimi

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

REBALANCE_FREQ = "ME"          # ayın son günü
DIVIDEND_YIELD_ANNUAL = 0.04   # temettü hisse ek getirisi varsayımı


# ─────────────────────────────────────────────────────────────────────────
# 1. Getiri serilerini oluştur
# ─────────────────────────────────────────────────────────────────────────
def build_return_series() -> pd.DataFrame:
    df = pd.read_csv(de.MERGED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    r = pd.DataFrame(index=df.index)
    r["mevduat"] = (1 + df["deposit_rate"] / 100) ** (1 / 365) - 1
    r["tahvil"]  = (1 + df.get("turkey_10y_bond", df["deposit_rate"]) / 100) ** (1 / 365) - 1
    r["doviz"]   = df["usdtry"].pct_change(fill_method=None)
    r["altin"]   = (df["gold_ons_usd"] * df["usdtry"]).pct_change(fill_method=None)
    r["hisse"]   = df["bist100"].pct_change(fill_method=None)
    r["temettu_hisse"] = r["hisse"] + ((1 + DIVIDEND_YIELD_ANNUAL) ** (1 / 365) - 1)
    r["yatirim_fonu"]  = 0.5 * r["hisse"] + 0.5 * r["mevduat"]
    r["kripto"]  = (df["btc_usd"] * df["usdtry"]).pct_change(fill_method=None)

    return r[de.ASSETS].dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────
# 2. Simülasyon: aylık yeniden dengeleme + günlük getiri bileşimi
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
    return {
        "total_return_%": total_return * 100,
        "cagr_%": cagr * 100,
        "ann_vol_%": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_%": max_dd * 100,
        "calmar": calmar,
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. Ana analiz
# ─────────────────────────────────────────────────────────────────────────
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
    print("               NOT: TR mevduat faizi bu dönemde %30-82 bandında gezdi —")
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
        for k in ["total_return_%", "cagr_%", "ann_vol_%", "sharpe", "max_drawdown_%", "calmar"]:
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
