from __future__ import annotations
"""
FAZ 3 — Kısıt-Native Optimizasyon Motoru (v7, deneysel).

Mevcut (v6) mekanizma: skor -> eşik-bazlı ayrık ±5/10/20pt ayar -> base'e
uygula -> 0'a klip -> [lo,hi] sınırına klip -> bütçe-nötr dengele (FIX-8) ->
sınır-korumalı su-doldurma normalizasyonu (FIX-9). Bu, elle tasarlanmış,
adım adım bir sezgisel akış — skorun BÜYÜKLÜĞÜNÜ (sadece hangi eşik
bandına düştüğünü) tam kullanmıyor, ve varlıklar arası KORELASYONU hiç
görmüyor (iki pozitif korelasyonlu varlığın aynı anda büyütülmesiyle
gerçek çeşitlendirme kaybı oluşabilir — kodun kendi eski TODO'su da
buna işaret ediyordu).

v7: klasik ortalama-varyans (mean-variance) optimizasyonu.
  - Beklenen getiri sinyali: FIX-10 skorlarından türetilir (score=0.5 -> 0,
    score=1.0 -> +mu_scale yıllık beklenen getiri sinyali).
  - Risk: backtest/simulate.py'nin getiri serilerinden hesaplanan GERÇEK
    yıllıklandırılmış kovaryans matrisi (varlıklar arası korelasyon dahil).
  - Kısıtlar scipy.optimize.minimize'a NATIVE verilir (bounds=, constraints=)
    — post-hoc klip/su-doldurma yamasına gerek YOK, çözüm zaten sınırları
    ve toplam=100 kısıtını sağlıyor.
  - Opsiyonel turnover cezası: önceki ay elde tutulan ağırlıktan sapmayı
    hafifçe cezalar (gerçek yeniden dengeleme maliyetini yaklaşıklar).
  - Kriz modu (CRISIS_OVERRIDE) AYNEN korunur — optimizer bypass edilir.

Geriye dönük uyumluluk: mevcut run_engine() / API / frontend sözleşmesi
DEĞİŞMEDİ. Bu modül run_engine_v7() adında AYRI bir giriş noktası sağlıyor;
v6 varsayılan ve canlı sistemde kullanılan sürüm olmaya devam ediyor.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import decision_engine as de
from backtest import build_return_series

MU_SCALE = 0.30           # score sapması -> yıllık beklenen getiri sinyali ölçeği
BASE_RISK_AVERSION = 4.0  # PROFILE_SENSITIVITY=1.0 (orta_riskli) için referans risk iştahsızlığı
TURNOVER_PENALTY = 0.0    # varsayılan: kapalı (tek seferlik run_engine çağrısı önceki ağırlığı bilmiyor)

_COV_CACHE: Optional[pd.DataFrame] = None


def get_covariance_matrix() -> pd.DataFrame:
    """Yıllıklandırılmış kovaryans matrisi — backtest getiri serilerinden, cache'lenir."""
    global _COV_CACHE
    if _COV_CACHE is None:
        returns = build_return_series()
        _COV_CACHE = returns.cov() * 252
    return _COV_CACHE


def expected_returns_from_scores(scores: Dict[str, float], mu_scale: float = MU_SCALE) -> np.ndarray:
    """score (0-1, 0.5=nötr) -> yıllık beklenen getiri sinyali (decimal)."""
    return np.array([(scores[a] - 0.5) * 2 * mu_scale for a in de.ASSETS])


def risk_aversion_from_sensitivity(sensitivity: float, base: float = BASE_RISK_AVERSION) -> float:
    """PROFILE_SENSITIVITY yüksek (agresif profil) -> risk iştahsızlığı düşük."""
    return base / sensitivity


def optimize_portfolio(scores: Dict[str, float], sensitivity: float,
                        bounds: Dict[str, tuple] = None,
                        prev_weights: Optional[Dict[str, float]] = None,
                        turnover_penalty: float = TURNOVER_PENALTY,
                        mu_scale: float = MU_SCALE,
                        target: float = 100.0) -> Dict[str, float]:
    """
    Kısıt-native mean-variance optimizasyonu. Döner: {varlık: yüzde}.
    Sınırlar scipy'ye native verilir; sonrası sadece yuvarlama artığı için
    de._normalize_within_bounds ile (zaten test edilmiş, bound-safe) temizlenir.
    """
    bounds = bounds or de.ASSET_BOUNDS
    assets = de.ASSETS
    n = len(assets)

    cov = get_covariance_matrix()
    Sigma = cov.loc[assets, assets].values
    mu = expected_returns_from_scores(scores, mu_scale=mu_scale)
    risk_aversion = risk_aversion_from_sensitivity(sensitivity)

    lo_hi = [bounds.get(a, (0.0, 100.0)) for a in assets]
    w0 = np.array([max(lo, min(hi, 100.0 / n)) for lo, hi in lo_hi])
    # feasibility: eşit dağılım sınırları ihlal ediyorsa ortasına çek
    w0 = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(w0, lo_hi)])

    prev = np.array([prev_weights.get(a, w0[i]) if prev_weights else w0[i]
                      for i, a in enumerate(assets)])

    def objective(w_pct: np.ndarray) -> float:
        w = w_pct / 100.0
        expected_ret = float(mu @ w)
        risk = float(w @ Sigma @ w)
        cost = turnover_penalty * float(np.sum((w_pct - prev) ** 2)) / 10_000.0
        return -(expected_ret - risk_aversion * risk) + cost

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - target}]

    result = minimize(objective, w0, method="SLSQP", bounds=lo_hi,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})

    if not result.success:
        # Emniyet: optimizer yakınsamazsa eşit-ağırlıklı (sınır içi) başlangıca düş.
        raw = {a: float(w0[i]) for i, a in enumerate(assets)}
    else:
        raw = {a: float(result.x[i]) for i, a in enumerate(assets)}

    return de._normalize_within_bounds(raw, bounds, target=target)


# ─────────────────────────────────────────────────────────────────────────
# run_engine_v7 — v6 ile aynı sözleşme (aynı dönüş şeması), sadece
# apply_adjustments yerine optimize_portfolio kullanır.
# ─────────────────────────────────────────────────────────────────────────
def run_engine_v7(profile: str, date: Optional[str] = None, amount: float = 100_000,
                   prev_weights: Optional[Dict[str, float]] = None,
                   turnover_penalty: float = TURNOVER_PENALTY) -> dict:
    if profile not in de.BASE_PORTFOLIOS:
        raise ValueError(f"Geçersiz profil: {profile}. Seçenekler: {list(de.BASE_PORTFOLIOS)}")

    df, eval_date = de.load_data(date)
    df = de.compute_derived(df)
    eval_row = df.iloc[-1]

    is_crisis, crisis_reason = de.check_crisis(eval_row)
    risk_score = float(eval_row.get("risk_score", 0.5))

    if is_crisis:
        final_alloc = de.CRISIS_OVERRIDE.copy()
        scores = {a: 0.5 for a in de.ASSETS}
        mode = "KRİZ"
    else:
        scores = de.compute_asset_scores(df, eval_row)
        sensitivity = de.PROFILE_SENSITIVITY[profile]
        final_alloc = optimize_portfolio(scores, sensitivity, prev_weights=prev_weights,
                                          turnover_penalty=turnover_penalty)
        mode = "NORMAL"

    tl_alloc = {k: round(amount * v / 100, 2) for k, v in final_alloc.items()}

    macro_cols = ["policy_rate", "real_rate", "usdtry", "usdtry_chg_30d",
                  "cpi_yoy", "bist_momentum", "fx_stress", "gold_real_return",
                  "vix_level", "cds_level", "risk_score"]
    macro_snapshot = {}
    for col in macro_cols:
        val = eval_row.get(col, np.nan)
        macro_snapshot[col] = round(float(val), 2) if pd.notna(val) else None

    signals = de._generate_signals(macro_snapshot, scores, is_crisis, crisis_reason)

    return {
        "engine_version": "v7",
        "eval_date": eval_date,
        "profile": profile,
        "mode": mode,
        "risk_score": round(risk_score, 3),
        "is_crisis": is_crisis,
        "crisis_reason": crisis_reason,
        "amount_tl": amount,
        "base_alloc": de.BASE_PORTFOLIOS[profile],
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "final_alloc": final_alloc,
        "tl_alloc": tl_alloc,
        "macro": macro_snapshot,
        "signals": signals,
        "is_backtest": date is not None,
        "requested_date": date,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(de.BASE_PORTFOLIOS), default="orta_riskli")
    parser.add_argument("--date", default=None)
    parser.add_argument("--amount", type=float, default=100_000)
    args = parser.parse_args()

    r = run_engine_v7(args.profile, date=args.date, amount=args.amount)
    print(f"v7 — {r['eval_date']} — {r['profile']} — mod={r['mode']} — risk={r['risk_score']}")
    print(f"toplam: {sum(r['final_alloc'].values()):.1f}%")
    for a in de.ASSETS:
        print(f"  {a:<16} {r['final_alloc'][a]:>6.1f}%   skor={r['scores'][a]:.3f}")
