from __future__ import annotations
"""
FAZ 4 — Kapsamlı Kalite/Değerlendirme Çerçevesi.

Herhangi bir motor sürümü (v6 varsayılan üretim motoru, v7 deneysel optimizer)
için standart, tekrarlanabilir bir "karne" üretir:
  - Performans metrikleri: CAGR, Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR %95
  - Benchmark karşılaştırması: statik taban portföy + %100 mevduat
  - Turnover: ortalama aylık |ağırlık değişimi|
  - Rejim ayrıştırma: motorun KRİZ dediği aylar vs NORMAL aylar ayrı raporlanır
  - Kararlılık kontrolü: backtest dönemi ikiye bölünüp (ilk yarı/ikinci yarı)
    Sharpe farkı raporlanır — gerçek bir train/test fit'i olmayan (v6/v7 kural
    bazlı) motorlar için "performansın tek bir alt-döneme sıkışmış olup
    olmadığını" gösteren jenerik bir sağlamlık göstergesi.
  - Sinyal sağlığı: signal_validation.py'nin IC testi yeniden çalıştırılır,
    kaç sinyalin hâlâ anlamlı (TUT) olduğu özetlenir (concept drift erken uyarısı).
  - İSTATİSTİKSEL ANLAMLILIK (FAZ 4.1): "Motor A, Motor B'yi geçti" iddiası
    SADECE nokta tahminine bakılarak (CAGR/Sharpe daha yüksek mi) VERİLMİYOR.
    Aylık getiri FARKI serisine eşleştirilmiş t-testi + bootstrap %95 GA
    uygulanıyor. Sebep: ~79 aylık gözlemle, GEÇTİ/GEÇEMEDİ gibi ikili bir
    karar gürültüyü gerçek etkiden ayıramayabiliyor — ilk sürümde tam bu
    hataya düşüldü (bkz. örnek: "0/3 profil statik tabanı geçti" sonucu,
    istatistiksel teste tabi tutulunca 3 profilin 3'ünde de p>0.19 ve
    bootstrap GA sıfırı içeriyor çıktı — yani "geçemedi" iddiası gürültüden
    ayrıştırılamıyordu). Artık her karşılaştırma p-değeri VE güven aralığıyla
    birlikte raporlanıyor; sadece p<0.05 VE GA sıfırı içermiyorsa "ANLAMLI
    GEÇTİ/KAYBETTİ" denir, aksi halde "FARK YOK (gürültüden ayrışmıyor)".

BU FAZDAN SONRAKİ KURAL: yeni bir motor sürümü/ağırlık seti, bu karneyi
göstermeden ve mevcut üretim motorunu (v6) İSTATİSTİKSEL OLARAK ANLAMLI
ŞEKİLDE geçmeden "üretime" alınamaz — nokta tahmini üstünlüğü yetmez.

Çalıştır:
  python quality_report.py --engine v6
  python quality_report.py --engine v7 --profile orta_riskli
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
N_BOOTSTRAP = 5000

import decision_engine as de
from backtest import build_return_series, simulate, compute_metrics

ALL_PROFILES = list(de.BASE_PORTFOLIOS.keys())


def get_engine_fn(engine: str):
    if engine == "v6":
        return de.run_engine
    elif engine == "v7":
        import optimizer as opt
        return opt.run_engine_v7
    raise ValueError(f"Bilinmeyen motor: {engine}")


def get_month_ends() -> list:
    all_dates = de.get_available_dates()
    all_dates_ts = pd.to_datetime(all_dates)
    month_ends = pd.Series(all_dates_ts).groupby(pd.Series(all_dates_ts).dt.to_period("M")).max().tolist()
    return [d for d in month_ends if d >= all_dates_ts.min() + pd.Timedelta(days=260)]


def run_backtest(engine_fn, profile: str, month_ends: list) -> tuple[dict, dict]:
    weights_at, mode_at = {}, {}
    for d in month_ends:
        r = engine_fn(profile, date=d.strftime("%Y-%m-%d"))
        weights_at[d] = r["final_alloc"]
        mode_at[d] = r["mode"]
    return weights_at, mode_at


def regime_breakdown(curve: pd.Series, month_ends: list, mode_at: dict) -> dict:
    """Her ay-sonu tarihindeki getiriyi, o tarihte motorun bildirdiği moda
    (KRİZ/NORMAL) göre kovalar. AYNI month_ends listesini kullanır (resample
    ile ayrı bir takvim üretmez) — böylece tarih hizalama sorunu olmaz."""
    vals = curve.reindex(month_ends, method="nearest")
    rets = vals.pct_change()   # rets.iloc[0] NaN (ilk ay için önceki değer yok)

    def bucket_stats(target_dates):
        r = [rets.loc[d] for d in target_dates if d in rets.index and pd.notna(rets.loc[d])]
        return {"n_months": len(target_dates), "avg_monthly_return_%": (np.mean(r) * 100 if r else float("nan"))}

    crisis_months = [d for d in month_ends if mode_at.get(d) == "KRİZ"]
    normal_months = [d for d in month_ends if mode_at.get(d) == "NORMAL"]
    return {"crisis": bucket_stats(crisis_months), "normal": bucket_stats(normal_months)}


def significance_test(curve_a: pd.Series, curve_b: pd.Series) -> dict:
    """
    A'nın B'yi GERÇEKTEN geçip geçmediğini test eder — nokta tahmini değil.
    Aylık getiri farkına (eşleştirilmiş, aynı ay/aynı piyasa gerçekleşmesi)
    tek-örneklem t-testi + bootstrap %95 GA uygulanır. p<0.05 VE GA sıfırı
    içermiyorsa "ANLAMLI"; aksi halde fark gürültüden ayrışmıyor demektir.
    """
    ret_a = curve_a.resample("ME").last().pct_change().dropna()
    ret_b = curve_b.resample("ME").last().pct_change().dropna()
    common = ret_a.index.intersection(ret_b.index)
    diff = (ret_a[common] - ret_b[common]) * 100
    if len(diff) < 10:
        return {"n": len(diff), "mean_diff_pt": float("nan"), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "significant": False}

    t_stat, p_val = stats.ttest_1samp(diff, 0)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    significant = bool(p_val < 0.05 and (ci_lo > 0 or ci_hi < 0))
    return {"n": len(diff), "mean_diff_pt": float(diff.mean()), "p_value": float(p_val),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "significant": significant}


def format_verdict(sig: dict, label_a: str, label_b: str) -> str:
    if sig["n"] < 10:
        return "YETERSİZ VERİ"
    direction = f"{label_a} > {label_b}" if sig["mean_diff_pt"] > 0 else f"{label_b} > {label_a}"
    if sig["significant"]:
        return f"ANLAMLI: {direction} (p={sig['p_value']:.3f}, GA=[{sig['ci_lo']:+.2f},{sig['ci_hi']:+.2f}]pt)"
    return (f"FARK YOK — gürültüden ayrışmıyor (p={sig['p_value']:.3f}, "
            f"GA=[{sig['ci_lo']:+.2f},{sig['ci_hi']:+.2f}]pt sıfırı içeriyor)")


def stability_check(curve: pd.Series, avg_deposit_rate: float) -> dict:
    mid = len(curve) // 2
    first_half = curve.iloc[:mid] / curve.iloc[0]
    second_half = curve.iloc[mid:] / curve.iloc[mid]
    m1 = compute_metrics(first_half, avg_deposit_rate)
    m2 = compute_metrics(second_half, avg_deposit_rate)
    return {"first_half_sharpe": m1["sharpe"], "second_half_sharpe": m2["sharpe"],
            "gap": abs(m1["sharpe"] - m2["sharpe"])}


def compute_turnover(weights_at: dict, month_ends: list) -> float:
    df = pd.DataFrame({d: weights_at[d] for d in month_ends}).T
    return float(df.diff().abs().sum(axis=1).mean())


def generate_report(engine: str, profiles: list) -> dict:
    print("=" * 90)
    print(f"  FAZ 4 — KALİTE KARNESİ — motor: {engine}")
    print("=" * 90)

    engine_fn = get_engine_fn(engine)
    returns = build_return_series()
    avg_deposit_rate = pd.read_csv(de.MERGED_CSV, parse_dates=["date"])["deposit_rate"].mean() / 100
    month_ends = get_month_ends()
    print(f"\nBacktest dönemi: {month_ends[0].date()} -> {month_ends[-1].date()} ({len(month_ends)} ay)")

    all_results = {}
    for profile in profiles:
        print(f"\n{'-'*90}\n  PROFİL: {profile.upper()}\n{'-'*90}")

        weights_at, mode_at = run_backtest(engine_fn, profile, month_ends)
        curve = simulate(returns, month_ends, weights_at)
        static_curve = simulate(returns, month_ends, {d: de.BASE_PORTFOLIOS[profile] for d in month_ends})
        cash_curve = simulate(returns, month_ends, {d: {"mevduat": 100} for d in month_ends})

        m = compute_metrics(curve, avg_deposit_rate)
        static_m = compute_metrics(static_curve, avg_deposit_rate)
        cash_m = compute_metrics(cash_curve, avg_deposit_rate)

        print(f"\n{'Metrik':<20}{'Motor':>12}{'Statik Taban':>14}{'%100 Mevduat':>14}")
        for k in ["total_return_%", "cagr_%", "ann_vol_%", "sharpe", "sortino",
                  "max_drawdown_%", "calmar", "var_95_%", "cvar_95_%"]:
            print(f"{k:<20}{m[k]:>12.2f}{static_m[k]:>14.2f}{cash_m[k]:>14.2f}")

        turnover = compute_turnover(weights_at, month_ends)
        print(f"\nOrtalama aylık turnover: {turnover:.1f}pt")

        regime = regime_breakdown(curve, month_ends, mode_at)
        print(f"\nRejim ayrıştırma:")
        print(f"  KRİZ ayları   : {regime['crisis']['n_months']} ay, "
              f"ort. aylık getiri = %{regime['crisis']['avg_monthly_return_%']:.2f}")
        print(f"  NORMAL aylar  : {regime['normal']['n_months']} ay, "
              f"ort. aylık getiri = %{regime['normal']['avg_monthly_return_%']:.2f}")

        stability = stability_check(curve, avg_deposit_rate)
        print(f"\nKararlılık: ilk yarı Sharpe={stability['first_half_sharpe']:.2f}, "
              f"ikinci yarı Sharpe={stability['second_half_sharpe']:.2f}, "
              f"fark={stability['gap']:.2f}"
              f"{'  [UYARI: büyük fark, tek döneme sıkışmış olabilir]' if stability['gap'] > 1.5 else ''}")

        sig_static = significance_test(curve, static_curve)
        sig_cash = significance_test(curve, cash_curve)
        print(f"\nİstatistiksel anlamlılık (n={sig_static['n']} ay eşleştirilmiş fark):")
        print(f"  Motor vs Statik Taban: {format_verdict(sig_static, 'Motor', 'Statik')}")
        print(f"  Motor vs %100 Mevduat: {format_verdict(sig_cash, 'Motor', 'Mevduat')}")

        all_results[profile] = {"engine": m, "static": static_m, "cash": cash_m,
                                 "turnover": turnover, "regime": regime, "stability": stability,
                                 "sig_static": sig_static, "sig_cash": sig_cash}

    # ── Sinyal sağlığı ──────────────────────────────────────────────────
    print(f"\n{'='*90}\n  SİNYAL SAĞLIĞI (signal_validation.py yeniden çalıştırıldı)\n{'='*90}")
    try:
        import signal_validation as sv
        sv_result = sv.run_validation()
        n_tut = (sv_result["karar"] == "TUT").sum()
        n_at = (sv_result["karar"] == "AT").sum()
        print(f"\n[quality_report özet] {n_tut}/{len(sv_result)} sinyal-varlık testi hâlâ anlamlı (TUT).")
        if n_at / max(len(sv_result), 1) > 0.3:
            print("[UYARI] Anlamsız sinyal oranı %30'u aşıyor — concept drift olasılığı, Faz 1'i tekrar gözden geçir.")
    except Exception as e:
        print(f"Sinyal sağlığı testi çalıştırılamadı: {e}")

    print(f"\n{'='*90}\n  GENEL SONUÇ (istatistiksel olarak anlamlı farklar)\n{'='*90}")
    n_sig_beat_static = sum(1 for p in all_results if all_results[p]["sig_static"]["significant"]
                             and all_results[p]["sig_static"]["mean_diff_pt"] > 0)
    n_sig_lose_static = sum(1 for p in all_results if all_results[p]["sig_static"]["significant"]
                             and all_results[p]["sig_static"]["mean_diff_pt"] < 0)
    n_no_diff_static = len(profiles) - n_sig_beat_static - n_sig_lose_static
    print(f"Statik tabana karşı: {n_sig_beat_static}/{len(profiles)} anlamlı ÜSTÜN, "
          f"{n_sig_lose_static}/{len(profiles)} anlamlı GERİDE, "
          f"{n_no_diff_static}/{len(profiles)} FARK YOK (gürültüden ayrışmıyor)")
    if n_no_diff_static > 0:
        print(f"NOT: 'fark yok' çıkan profiller için ne 'motor daha iyi' ne 'statik daha iyi' iddia edilebilir —")
        print(f"     mevcut örneklem büyüklüğüyle (~{month_ends and len(month_ends)} ay) ayrıştırılamıyor.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["v6", "v7"], default="v6")
    parser.add_argument("--profile", choices=ALL_PROFILES + ["all"], default="all")
    args = parser.parse_args()

    profiles = ALL_PROFILES if args.profile == "all" else [args.profile]
    generate_report(args.engine, profiles)
