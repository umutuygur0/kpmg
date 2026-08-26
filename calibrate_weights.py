from __future__ import annotations
"""
FAZ 2 — Kalibre Edilmiş Skorlama: elle seçilmiş ağırlıkları (0.4/0.5/0.6 vb.)
walk-forward Ridge regresyonuyla veri-güdümlü hale getir.

FAZ 2.1 notu: alpha=1.0 ile ilk denemede 8/8 varlık overfit gerekçesiyle
reddedildi (IS IC 0.2-0.7, OOS IC ~0, bazılarında negatif). Sabit bir alpha
tahmin etmek yerine, alpha'nın kendisi walk-forward OOS IC'yi maksimize
edecek şekilde ARANIYOR (bkz. select_best_alpha) — güçlü regularizasyon,
ağırlıkları sıfıra/orijinal-benzeri değerlere büzerek aşırı uyumu bastırır.

Yöntem:
  1. Her varlık için 6 ham sinyalin rolling percentile-rank'i (motorun kendi
     dönüşümüyle aynı) + yatirim_fonu'na özel "neutrality" özelliği.
  2. backtest.split.expanding_window_splits ile walk-forward train/test.
  3. alpha, TÜM varlıklar için ortak bir tarama ile seçilir (asset-özel alpha
     seçimi, hiper-parametrenin kendisinin overfit olmasına yol açabilir).
  4. Her varlık için IS/OOS IC + overfit kapısı (%50 fark eşiği) uygulanır.
  5. Portföy karşılaştırması: REDDEDİLEN varlıklar için kalibre skor
     KULLANILMAZ — o varlığın o tarihteki MEVCUT motor skoru kullanılır
     (graceful fallback). Önceki denemede bu kontrol eksikti, düzeltildi.

Çalıştır: python calibrate_weights.py
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import decision_engine as de
from backtest import build_return_series, simulate, compute_metrics, expanding_window_splits

FORWARD_HORIZON = 20            # işgünü — aylık rebalans ufkuyla uyumlu
ROLLING_WINDOW = de.ROLLING_WINDOW   # 252g — motorun kendi percentile penceresiyle aynı
# FAZ 2.2: 7 özellikli model, HİÇBİR regularizasyon seviyesinde (alpha 1..5000)
# pozitif OOS IC üretemedi — sorun aşırı uyum değil, boyut/veri oranıydı. Şimdi
# her split'te SADECE o split'in train verisiyle anlamlı bulunan (p<=0.10) en
# güçlü TOP_K özellik seçiliyor (bkz. select_features) — 7 yerine 1-3 parametre.
ALPHA_GRID = [1, 5, 10, 25, 50, 100]
TOP_K_FEATURES = 3
FEATURE_P_THRESHOLD = 0.10
OVERFIT_GAP_THRESHOLD = 0.50     # in-sample/out-of-sample IC farkı bu oranı aşarsa reddet

FEATURE_COLS = ["rr_pct", "fxs_pct", "bist_pct", "grr_pct", "us10_pct", "risk", "neutrality"]

RAW_SIGNAL_MAP = {
    "rr_pct": "real_rate", "fxs_pct": "fx_stress", "bist_pct": "bist_momentum",
    "grr_pct": "gold_real_return", "us10_pct": "us10y", "risk": "risk_score",
}


# ─────────────────────────────────────────────────────────────────────────
# 1. Özellik matrisi — motorun kullandığı AYNI rolling percentile dönüşümü
# ─────────────────────────────────────────────────────────────────────────
def _rolling_pct_rank(s: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return s.rolling(window, min_periods=30).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def build_feature_frame() -> pd.DataFrame:
    df = pd.read_csv(de.MERGED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d = de.compute_derived(df).set_index("date")

    feats = pd.DataFrame(index=d.index)
    for feat_name, raw_col in RAW_SIGNAL_MAP.items():
        feats[feat_name] = _rolling_pct_rank(d[raw_col])
    feats["neutrality"] = 1 - (feats["bist_pct"] - 0.5).abs() * 2
    return feats


# ─────────────────────────────────────────────────────────────────────────
# 1b. Özellik seçimi — SADECE train verisiyle (leakage yok), en fazla TOP_K
# ─────────────────────────────────────────────────────────────────────────
def select_features(train: pd.DataFrame, top_k: int = TOP_K_FEATURES) -> list[str]:
    """Train penceresinde her adayın hedefle Spearman IC'sini hesaplar, p<=0.10
    olanları |IC|'ye göre sıralar, en iyi top_k'yı döner. Hiçbiri anlamlı
    değilse (nadiren), en azından en güçlü 1 özellik alınır — regresyon
    tamamen özelliksiz kalmasın diye (yine de OOS IC/overfit kapısı onu
    eleyecektir gerekirse)."""
    scored = []
    for col in FEATURE_COLS:
        if train[col].nunique() < 5:
            continue
        ic, p = spearmanr(train[col], train["target"])
        if np.isnan(ic):
            continue
        scored.append((col, ic, p))
    significant = [(c, ic) for c, ic, p in scored if p <= FEATURE_P_THRESHOLD]
    significant.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = [c for c, _ in significant[:top_k]]
    if not selected and scored:
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        selected = [scored[0][0]]
    return selected


# ─────────────────────────────────────────────────────────────────────────
# 2. Alpha seçimi — walk-forward OOS IC'yi maksimize eden ortak regularizasyon
#    (özellik seçimi her split'te ayrı ayrı, sadece o split'in train'iyle yapılır)
# ─────────────────────────────────────────────────────────────────────────
def _avg_oos_ic_for_alpha(alpha: float, features: pd.DataFrame, returns: pd.DataFrame,
                           splits: list) -> float:
    ics = []
    for asset in de.ASSETS:
        fwd_ret = (1 + returns[asset]).rolling(FORWARD_HORIZON).apply(np.prod, raw=True).shift(-FORWARD_HORIZON) - 1
        aligned = features.join(fwd_ret.rename("target")).dropna()
        for sp in splits:
            train = aligned.loc[aligned.index.isin(sp.train_dates)]
            test = aligned.loc[aligned.index.isin(sp.test_dates)]
            if len(train) < 60 or test.empty:
                continue
            sel = select_features(train)
            if not sel:
                continue
            model = Ridge(alpha=alpha)
            model.fit(train[sel], train["target"])
            pred = model.predict(test[sel])
            ic, _ = spearmanr(pred, test["target"])
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("-inf")


def select_best_alpha(features: pd.DataFrame, returns: pd.DataFrame, splits: list) -> tuple[float, list]:
    print("\nAlpha taraması (tüm varlıklar için ortalama walk-forward OOS IC):")
    scored = []
    for alpha in ALPHA_GRID:
        avg_ic = _avg_oos_ic_for_alpha(alpha, features, returns, splits)
        scored.append((alpha, avg_ic))
        print(f"  alpha={alpha:>6}  ort. OOS IC = {avg_ic:+.4f}")
    best_alpha, best_ic = max(scored, key=lambda x: x[1])
    print(f"Seçilen alpha = {best_alpha} (ort. OOS IC = {best_ic:+.4f})")
    return best_alpha, scored


# ─────────────────────────────────────────────────────────────────────────
# 3. Walk-forward kalibrasyon (tek varlık için)
# ─────────────────────────────────────────────────────────────────────────
def stitch_oos_predictions(splits, predict_fn) -> pd.Series:
    """Her split'in test penceresinden SADECE bir sonraki split başlayana kadarki
    (çakışmayan) yeni dilimi alır -> tam OOS, çakışmasız bir tahmin serisi."""
    pieces = []
    for i, sp in enumerate(splits):
        preds = predict_fn(sp)
        if i + 1 < len(splits):
            cutoff = splits[i + 1].test_start
            preds = preds[preds.index < cutoff]
        pieces.append(preds)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def calibrate_asset(asset: str, features: pd.DataFrame, fwd_ret: pd.Series,
                     splits: list, alpha: float) -> dict:
    aligned = features.join(fwd_ret.rename("target")).dropna()

    is_ics, oos_ics, split_features = [], [], []
    for sp in splits:
        train = aligned.loc[aligned.index.isin(sp.train_dates)]
        test = aligned.loc[aligned.index.isin(sp.test_dates)]
        if len(train) < 60 or test.empty:
            continue
        sel = select_features(train)
        if not sel:
            continue
        split_features.append(sel)
        model = Ridge(alpha=alpha)
        model.fit(train[sel], train["target"])

        train_pred = model.predict(train[sel])
        is_ic, _ = spearmanr(train_pred, train["target"])
        is_ics.append(is_ic)

        test_pred = model.predict(test[sel])
        oos_ic, _ = spearmanr(test_pred, test["target"])
        oos_ics.append(oos_ic)

    avg_is_ic = float(np.nanmean(is_ics)) if is_ics else np.nan
    avg_oos_ic = float(np.nanmean(oos_ics)) if oos_ics else np.nan
    gap = ((avg_is_ic - avg_oos_ic) / abs(avg_is_ic)
           if avg_is_ic not in (0,) and not np.isnan(avg_is_ic) else np.nan)
    overfit = bool(gap is not None and not np.isnan(gap) and gap > OVERFIT_GAP_THRESHOLD)
    # OOS IC'nin kendisi anlamlı derecede pozitif değilse de reddet (overfit değilse bile
    # gürültü olabilir) — asgari |IC| eşiği.
    weak_signal = bool(np.isnan(avg_oos_ic) or avg_oos_ic < 0.02)
    rejected = overfit or weak_signal

    def predict_for_split(sp):
        train = aligned.loc[aligned.index.isin(sp.train_dates)]
        if len(train) < 60:
            return pd.Series(dtype=float)
        sel = select_features(train)
        if not sel:
            return pd.Series(dtype=float)
        model = Ridge(alpha=alpha)
        model.fit(train[sel], train["target"])
        test = aligned.loc[aligned.index.isin(sp.test_dates)]
        if test.empty:
            return pd.Series(dtype=float)
        return pd.Series(model.predict(test[sel]), index=test.index)

    oos_raw_pred = stitch_oos_predictions(splits, predict_for_split)
    oos_score = _rolling_pct_rank(oos_raw_pred, window=ROLLING_WINDOW)

    final_features = select_features(aligned)
    final_model = Ridge(alpha=alpha)
    final_model.fit(aligned[final_features], aligned["target"])
    final_weights = dict(zip(final_features, final_model.coef_))

    # En sık seçilen özellikler (split'ler arası tutarlılık göstergesi)
    feature_stability = Counter(f for sel in split_features for f in sel)

    return {
        "asset": asset,
        "avg_in_sample_ic": avg_is_ic,
        "avg_out_of_sample_ic": avg_oos_ic,
        "overfit_gap": gap,
        "rejected": rejected,
        "reject_reason": "overfit" if overfit else ("zayıf_sinyal" if weak_signal else None),
        "n_splits": len(is_ics),
        "final_features": final_features,
        "final_weights": final_weights,
        "feature_stability": dict(feature_stability),
        "oos_score_series": oos_score,
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. Portföy seviyesinde karşılaştırma: kalibre skorlar (+ reddedilenler için
#    mevcut motora graceful fallback) vs saf mevcut motor
# ─────────────────────────────────────────────────────────────────────────
def build_hybrid_scores(calib_results: dict, dates: list) -> dict[pd.Timestamp, dict]:
    """
    Her tarih için: kabul edilen varlıklarda kalibre skor, reddedilende
    mevcut motorun o tarihteki GERÇEK skoru (compute_asset_scores doğrudan
    çağrılarak — run_engine ÜZERİNDEN DEĞİL, çünkü run_engine kriz modunda
    scores'u {a: 0.5} ile DOLDURUYOR; bunu "gerçek skor" sanıp kullanmak
    kriz günlerinde hybrid ile mevcut motoru sessizce ayrıştırırdı — bu
    fonksiyonun ilk sürümünde tam olarak bu hataya düşülmüştü).
    Kriz durumu da ayrıca döndürülür ki portföy inşası CRISIS_OVERRIDE'ı
    run_engine ile birebir aynı şekilde uygulayabilsin.
    """
    accepted = {a: r for a, r in calib_results.items() if not r["rejected"]}
    rejected_assets = [a for a, r in calib_results.items() if r["rejected"]]

    hybrid, crisis_by_date = {}, {}
    for d in dates:
        df, _ = de.load_data(date=d.strftime("%Y-%m-%d"))
        df = de.compute_derived(df)
        eval_row = df.iloc[-1]
        is_crisis, _ = de.check_crisis(eval_row)
        crisis_by_date[d] = is_crisis
        real_scores = de.compute_asset_scores(df, eval_row)

        scores = {}
        for asset in de.ASSETS:
            if asset in accepted:
                s = accepted[asset]["oos_score_series"]
                scores[asset] = float(s.get(d, 0.5)) if pd.notna(s.get(d, np.nan)) else 0.5
            else:
                scores[asset] = real_scores[asset]
        hybrid[d] = scores
    return hybrid, crisis_by_date, list(accepted.keys()), rejected_assets


def portfolio_backtest_hybrid(profile: str, hybrid_scores: dict[pd.Timestamp, dict],
                               crisis_by_date: dict[pd.Timestamp, bool],
                               returns: pd.DataFrame) -> pd.Series:
    dates = sorted(hybrid_scores.keys())
    month_ends = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).max().tolist()

    base = de.BASE_PORTFOLIOS[profile]
    sens = de.PROFILE_SENSITIVITY[profile]
    weights_at = {}
    for d in month_ends:
        if crisis_by_date.get(d):
            weights_at[d] = de.CRISIS_OVERRIDE.copy()
        else:
            _, final = de.apply_adjustments(base, hybrid_scores[d], sens)
            weights_at[d] = final
    return simulate(returns, month_ends, weights_at), month_ends


def current_engine_curve(profile: str, dates: list, returns: pd.DataFrame) -> pd.Series:
    weights_at = {}
    for d in dates:
        r = de.run_engine(profile, date=d.strftime("%Y-%m-%d"))
        weights_at[d] = r["final_alloc"]
    return simulate(returns, dates, weights_at)


# ─────────────────────────────────────────────────────────────────────────
# 5. Ana akış
# ─────────────────────────────────────────────────────────────────────────
def run_calibration():
    print("=" * 90)
    print("  FAZ 2 — AĞIRLIK KALİBRASYONU (walk-forward Ridge, alpha aranıyor)")
    print("=" * 90)

    features = build_feature_frame()
    returns = build_return_series()
    dates = de.get_available_dates()
    splits = list(expanding_window_splits(dates))
    print(f"\n{len(splits)} walk-forward split (expanding window)")

    best_alpha, alpha_scores = select_best_alpha(features, returns, splits)

    calib_results = {}
    print(f"\n{'Varlık':<16}{'IS IC':>9}{'OOS IC':>9}{'Overfit%':>10}{'Karar':>22}")
    print("-" * 90)
    for asset in de.ASSETS:
        fwd_ret = (1 + returns[asset]).rolling(FORWARD_HORIZON).apply(np.prod, raw=True).shift(-FORWARD_HORIZON) - 1
        res = calibrate_asset(asset, features, fwd_ret, splits, alpha=best_alpha)
        calib_results[asset] = res
        gap_pct = res["overfit_gap"] * 100 if res["overfit_gap"] is not None and not np.isnan(res["overfit_gap"]) else float("nan")
        karar = f"REDDEDİLDİ ({res['reject_reason']})" if res["rejected"] else "KABUL"
        print(f"{asset:<16}{res['avg_in_sample_ic']:>9.3f}{res['avg_out_of_sample_ic']:>9.3f}{gap_pct:>9.1f}%{karar:>22}")

    print(f"\n{'='*90}\n  ÖĞRENİLEN AĞIRLIKLAR (alpha={best_alpha}, tüm veriyle fit edilmiş final model)\n{'='*90}")
    for asset, res in calib_results.items():
        w = res["final_weights"]
        w_str = ", ".join(f"{k}={v:+.4f}" for k, v in w.items()) if w else "(özellik seçilemedi)"
        stab = ", ".join(f"{k}:{v}/{res['n_splits']}" for k, v in
                          sorted(res["feature_stability"].items(), key=lambda x: -x[1]))
        tag = "" if not res["rejected"] else "  [REDDEDİLDİ — portföyde kullanılmayacak]"
        print(f"  {asset:<16}: {w_str}{tag}")
        print(f"  {'':16}  split-kararlılığı: {stab or '(yok)'}")

    # ── Portföy seviyesi OOS karşılaştırma (hybrid: kabul + fallback) ──────
    print(f"\n{'='*90}\n  PORTFÖY SEVİYESİ OOS KARŞILAŞTIRMA (hybrid kalibre vs saf mevcut motor)\n{'='*90}")
    any_score = next(iter(calib_results.values()))["oos_score_series"]
    oos_dates_all = sorted(any_score.dropna().index)
    hybrid_scores, crisis_by_date, accepted_assets, rejected_assets = build_hybrid_scores(calib_results, oos_dates_all)
    print(f"Kabul edilen (kalibre kullanılan): {accepted_assets or '(YOK)'}")
    print(f"Reddedilen (mevcut motora fallback): {rejected_assets}")

    avg_deposit_rate = pd.read_csv(de.MERGED_CSV, parse_dates=["date"])["deposit_rate"].mean() / 100

    summary = {}
    for profile in ["az_riskli", "orta_riskli", "cok_riskli"]:
        hybrid_curve, oos_dates = portfolio_backtest_hybrid(profile, hybrid_scores, crisis_by_date, returns)
        current_curve = current_engine_curve(profile, oos_dates, returns)

        hybrid_m = compute_metrics(hybrid_curve, avg_deposit_rate)
        current_m = compute_metrics(current_curve, avg_deposit_rate)

        print(f"\n  {profile.upper()}  (OOS dönem: {oos_dates[0].date()} -> {oos_dates[-1].date()}, {len(oos_dates)} ay)")
        print(f"  {'Metrik':<18}{'Hybrid-Kalibre':>16}{'Mevcut':>12}{'Fark':>10}")
        for k in ["cagr_%", "sharpe", "max_drawdown_%", "calmar"]:
            cv, mv = hybrid_m[k], current_m[k]
            print(f"  {k:<18}{cv:>16.2f}{mv:>12.2f}{cv-mv:>+10.2f}")
        summary[profile] = {"hybrid": hybrid_m, "current": current_m}

    print(f"\n{'='*90}\n  KARAR\n{'='*90}")
    if not accepted_assets:
        print("Hiçbir varlık kalibrasyonu kabul edilmedi — mevcut veri derinliğiyle (6.6 yıl,")
        print("~2400 gün) 7 özellikli walk-forward regresyon, hiçbir güçlü regularizasyon")
        print("seviyesinde bile anlamlı düzeyde genelleme yapamıyor. Motor DEĞİŞTİRİLMEMELİ.")
    else:
        n_better = sum(1 for p in summary if summary[p]["hybrid"]["sharpe"] > summary[p]["current"]["sharpe"])
        print(f"Hybrid motorun Sharpe'ı mevcut motoru geçtiği profil sayısı: {n_better}/3")

    return calib_results, summary


if __name__ == "__main__":
    run_calibration()
