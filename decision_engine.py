from __future__ import annotations
"""
Portfolio Decision Engine — v6
Değişiklikler (v5 → v6):
  FIX-1  NEUTRAL_THRESHOLD 0.15 → 0.08  (nötr bölge kapanı giderildi)
  FIX-2  ASSET_BOUNDS eklendi           (varlık başına min/max sınır)
  FIX-3  apply_adjustments bounds uygular ve normalize eder
  FIX-4  percentile_rank → rolling_percentile_rank (252 günlük pencere)
  FIX-5  Kriz modu: risk_score eşiğine EK olarak VIX>40 veya CDS>700 hard-tetik
  FIX-6  compute_asset_scores profil argümanı KALDIRILDI (zaten profile bağlı değildi,
          ama dışarıdan çağrıldığında kafa karışıklığı yaratıyordu)

Kullanım:
  python decision_engine.py                       # interaktif
  python decision_engine.py --profile az_riskli
  python decision_engine.py --profile cok_riskli --date 2025-06-01 --save
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR      = Path(os.environ.get("OUTPUT_ROOT", "./portfolio_data"))
PROCESSED_DIR = BASE_DIR / "processed"
MERGED_CSV    = PROCESSED_DIR / "merged_aligned_daily.csv"
ENGINE_OUT    = BASE_DIR / "engine_output"
ENGINE_OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SABITLER
# =============================================================================
ASSETS = ["mevduat", "doviz", "altin", "tahvil",
          "yatirim_fonu", "hisse", "temettu_hisse", "kripto"]

ASSET_LABELS = {
    "mevduat":       "Mevduat / Para Fonu",
    "doviz":         "Döviz (USD/EUR)",
    "altin":         "Altın",
    "tahvil":        "Tahvil / Eurobond",
    "yatirim_fonu":  "Yatırım Fonları",
    "hisse":         "Hisse Senetleri",
    "temettu_hisse": "Temettü Hisseleri",
    "kripto":        "Kripto",
}

# ── Base portföyler ──────────────────────────────────────────────────────────
BASE_PORTFOLIOS: Dict[str, Dict[str, float]] = {
    "az_riskli": {
        "mevduat": 35, "doviz": 20, "altin": 25, "tahvil": 12,
        "yatirim_fonu": 8, "hisse": 0, "temettu_hisse": 0, "kripto": 0,
    },
    "orta_riskli": {
        "mevduat": 15, "doviz": 10, "altin": 20, "tahvil": 5,
        "yatirim_fonu": 10, "hisse": 25, "temettu_hisse": 15, "kripto": 0,
    },
    "cok_riskli": {
        "mevduat": 0, "doviz": 5, "altin": 10, "tahvil": 0,
        "yatirim_fonu": 10, "hisse": 40, "temettu_hisse": 20, "kripto": 15,
    },
}

CRISIS_OVERRIDE: Dict[str, float] = {
    "mevduat": 40, "doviz": 30, "altin": 25, "tahvil": 5,
    "yatirim_fonu": 0, "hisse": 0, "temettu_hisse": 0, "kripto": 0,
}

# ── FIX-2: Varlık başına min/max sınırlar (%pt) ──────────────────────────────
# Motor bu sınırları asla aşamaz, normalize edilse bile.
ASSET_BOUNDS: Dict[str, Tuple[float, float]] = {
    "mevduat":       (5,  55),   # min %5 güvenli liman, max %55
    "doviz":         (0,  35),   # tam çıkabilir, max %35
    "altin":         (5,  40),   # min %5 enflasyon koruması
    "tahvil":        (0,  25),
    "yatirim_fonu":  (0,  20),
    "hisse":         (0,  55),
    "temettu_hisse": (0,  30),
    "kripto":        (0,  15),   # max %15 — agresif profil base'i zaten %15
}

# ── Profil sensitivitesi ─────────────────────────────────────────────────────
PROFILE_SENSITIVITY = {
    "az_riskli":   0.6,
    "orta_riskli": 1.0,
    "cok_riskli":  1.4,
}

# ── FIX-1: Nötr bölge eşiği 0.15 → 0.08 ────────────────────────────────────
NEUTRAL_THRESHOLD = 0.08   # eski: 0.15

# Kriz soft eşiği (risk_score)
CRISIS_THRESHOLD = 0.80

# ── FIX-5: Hard kriz tetikleyiciler (anlık, risk_score'dan bağımsız) ─────────
VIX_CRISIS_LEVEL = 40.0
CDS_CRISIS_LEVEL = 700.0

# Ayarlama büyüklükleri (pt)
ADJ = {"hafif": 5, "orta": 10, "guclu": 20}

# Percentile rolling pencere (iş günü)
ROLLING_WINDOW = 252   # 1 yıl


# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================
def load_data(date: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if not MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Merged CSV bulunamadı: {MERGED_CSV}\n"
            "Önce portfolio_engine_v5.py çalıştırın."
        )
    df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if date:
        cutoff = pd.to_datetime(date)
        df = df[df["date"] <= cutoff]
        if df.empty:
            raise ValueError(f"Belirtilen tarihten önce veri yok: {date}")
    eval_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    return df, eval_date


# =============================================================================
# 2. DERIVED VARIABLES
# =============================================================================
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham veriden türetilmiş sinyal değişkenleri.
    Hiçbir profil bilgisi kullanmaz — sadece makro/piyasa verisi.
    """
    d = df.copy()

    # Enflasyon yıllık
    if "cpi_index" in d.columns:
        d["cpi_yoy"] = d["cpi_index"].pct_change(252) * 100
    else:
        d["cpi_yoy"] = np.nan

    # Reel faiz
    if "policy_rate" in d.columns and "cpi_yoy" in d.columns:
        d["real_rate"] = d["policy_rate"] - d["cpi_yoy"]
    else:
        d["real_rate"] = np.nan

    # FX Stress
    if "usdtry" in d.columns:
        d["usdtry_chg_30d"] = d["usdtry"].pct_change(30) * 100
        roll_std            = d["usdtry"].rolling(30).std()
        roll_mean           = d["usdtry"].rolling(30).mean()
        d["fx_stress"]      = (roll_std / roll_mean).clip(0, 1)
    else:
        d["fx_stress"]      = np.nan
        d["usdtry_chg_30d"] = np.nan

    # BIST Momentum
    if "bist100" in d.columns:
        ma20               = d["bist100"].rolling(20).mean()
        d["bist_momentum"] = ((d["bist100"] - ma20) / ma20 * 100).clip(-30, 30)
    else:
        d["bist_momentum"] = np.nan

    # Altın TRY reel getirisi
    if "gold_ons_usd" in d.columns and "usdtry" in d.columns:
        d["gold_try"]        = d["gold_ons_usd"] * d["usdtry"]
        d["gold_return_yoy"] = d["gold_try"].pct_change(252) * 100
        d["gold_real_return"] = (
            d["gold_return_yoy"] - d["cpi_yoy"]
            if "cpi_yoy" in d.columns
            else d["gold_return_yoy"]
        )
    else:
        d["gold_real_return"] = np.nan

    if "vix"          in d.columns: d["vix_level"] = d["vix"]
    else:                            d["vix_level"] = np.nan

    if "turkey_cds_5y" in d.columns: d["cds_level"] = d["turkey_cds_5y"]
    else:                             d["cds_level"] = np.nan

    # Bileşik Risk Skoru
    risk_components = {}
    if d["vix_level"].notna().any():
        risk_components["vix"] = (d["vix_level"].clip(10, 50) - 10) / 40
    if d["fx_stress"].notna().any():
        risk_components["fx"] = d["fx_stress"].clip(0, 1)
    if d["real_rate"].notna().any():
        risk_components["real_rate"] = (-d["real_rate"].clip(-50, 50) + 50) / 100
    if d["cds_level"].notna().any():
        risk_components["cds"] = (d["cds_level"].clip(200, 800) - 200) / 600
    if d["bist_momentum"].notna().any():
        risk_components["bist"] = (-d["bist_momentum"].clip(-30, 30) + 30) / 60

    weights = {"vix": 0.20, "fx": 0.25, "real_rate": 0.20, "cds": 0.20, "bist": 0.15}
    if risk_components:
        total_w, score = 0.0, pd.Series(0.0, index=d.index)
        for k, series in risk_components.items():
            w = weights.get(k, 0.2)
            score   += series.fillna(0.5) * w
            total_w += w
        d["risk_score"] = (score / total_w).clip(0, 1)
    else:
        d["risk_score"] = 0.5

    return d


# =============================================================================
# 3. FIX-4: ROLLING PERCENTILE RANK
# =============================================================================
def rolling_percentile_rank(series: pd.Series,
                             value: float,
                             window: int = ROLLING_WINDOW) -> float:
    """
    Değerin, son `window` gün içindeki dağılımdaki yüzdelik dilimi (0–1).
    Pencerede yeterli veri yoksa (< 30 gün) tüm seriyi kullanır.

    Eski `percentile_rank` → tüm historik veriyi kullanıyordu.
    Bu fonksiyon → sadece son ROLLING_WINDOW günü kullanır.
    Neden daha iyi: 2023'ün kriz ortamı 2026'nın normalini bozmaz.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return 0.5
    # Son window kadar al
    window_data = clean.iloc[-window:] if len(clean) >= window else clean
    if len(window_data) < 30:
        window_data = clean   # fallback: veri azsa hepsini kullan
    return float((window_data <= value).mean())


# Geriye dönük uyumluluk (stat_report.py için)
def percentile_rank(series: pd.Series, value: float) -> float:
    return rolling_percentile_rank(series, value)


# =============================================================================
# 4. VARLIK SKORLARI
# =============================================================================
def compute_asset_scores(df: pd.DataFrame,
                          eval_row: pd.Series) -> Dict[str, float]:
    """
    Her varlık için 0–1 arası cazibiyet skoru.
    PROFILE BAĞIMSIZ — sadece makro veriye bakar.
    Profil etkisi: apply_adjustments içindeki sensitivity çarpanında.

    Skor yüksekse: bu varlık şu an cazip → base'e pt eklenir
    Skor düşükse:  bu varlık şu an cazip değil → base'den pt düşülür
    """
    s: Dict[str, float] = {}

    def safe_pct(col: str, reverse: bool = False) -> float:
        if col not in df.columns or df[col].dropna().empty:
            return 0.5
        val = eval_row.get(col, np.nan)
        if pd.isna(val):
            return 0.5
        r = rolling_percentile_rank(df[col], float(val))
        return 1 - r if reverse else r

    risk    = float(eval_row.get("risk_score",    0.5) or 0.5)
    rr_pct  = safe_pct("real_rate")
    fxs_pct = safe_pct("fx_stress")
    bist_pct = safe_pct("bist_momentum")
    grr_pct  = safe_pct("gold_real_return")
    us10_inv = safe_pct("us10y", reverse=True)

    # Mevduat: reel faiz yüksekse + risk yüksekse (güvenli liman)
    s["mevduat"]       = rr_pct * 0.6  + risk * 0.4

    # Döviz: kur baskısı yüksekse + reel faiz düşükse
    s["doviz"]         = fxs_pct * 0.5 + risk * 0.3 + (1 - rr_pct) * 0.2

    # Altın: risk yüksekse + altın reel getirisi iyiyse
    s["altin"]         = risk * 0.5    + grr_pct * 0.3 + fxs_pct * 0.2

    # Tahvil: reel faiz pozitif + risk düşükse
    s["tahvil"]        = rr_pct * 0.5  + (1 - risk) * 0.3 + us10_inv * 0.2

    # Yatırım Fonu: piyasa ne çok iyi ne çok kötü (ortada) → karma fon cazip
    neutrality         = 1 - abs(bist_pct - 0.5) * 2
    s["yatirim_fonu"]  = neutrality * 0.4 + (1 - risk) * 0.3 + rr_pct * 0.3

    # Hisse: borsa momentum + düşük risk
    s["hisse"]         = bist_pct * 0.5 + (1 - risk) * 0.5

    # Temettü: faiz düşükse tahvil alternatifi olarak cazip + stabil piyasa
    s["temettu_hisse"] = (1 - rr_pct) * 0.4 + (1 - risk) * 0.4 + bist_pct * 0.2

    # Kripto: risk-on + global likidite yüksekse
    s["kripto"]        = (1 - risk) * 0.6 + us10_inv * 0.4

    for k in s:
        s[k] = float(np.clip(s[k], 0, 1))

    return s


# =============================================================================
# 5. FIX-1 + FIX-2: AYARLAMA MOTORU
# =============================================================================
def score_to_adjustment(score: float, sensitivity: float) -> float:
    """
    Skor → ±pt.
    FIX-1: NEUTRAL_THRESHOLD = 0.08 (eski: 0.15)
    Bu değişiklikle 8 varlıktan sadece 1-2'si nötr kalır,
    motor çok daha duyarlı hale gelir.
    """
    deviation = score - 0.5
    abs_dev   = abs(deviation)
    direction = 1 if deviation > 0 else -1

    if abs_dev < NEUTRAL_THRESHOLD:       # FIX-1: 0.08 (eski 0.15)
        raw_pt = 0
    elif abs_dev < 0.22:
        raw_pt = ADJ["hafif"]             # ±5 pt
    elif abs_dev < 0.38:
        raw_pt = ADJ["orta"]              # ±10 pt
    else:
        raw_pt = ADJ["guclu"]             # ±20 pt

    return direction * raw_pt * sensitivity


def apply_adjustments(base: Dict[str, float],
                       scores: Dict[str, float],
                       sensitivity: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    FIX-7: Base=0 varlıklara negatif ayar uygulanmaz.
    Döner: (adj_map, final_alloc) — ikisi de dışarıya çıkar.

    Neden önemli: base=0 bir varlık için negatif adj, normalize esnasında
    diğer varlıkların değerini bozar. Hisse=0 → adj=-5 → 0'a klip →
    toplam 95 kalır → normalize 100/95=1.05 ile çarpılır → her şey şişer.
    """
    adj_map = {}
    for asset in ASSETS:
        adj = score_to_adjustment(scores.get(asset, 0.5), sensitivity)
        b   = base.get(asset, 0)
        # FIX-7: base=0 olan varlığa negatif ayar yapma
        if b == 0 and adj < 0:
            adj = 0.0
        adj_map[asset] = adj

    adjusted = {asset: base.get(asset, 0) + adj_map[asset] for asset in ASSETS}

    for k in adjusted:
        adjusted[k] = max(0.0, adjusted[k])

    for k in adjusted:
        lo, hi = ASSET_BOUNDS.get(k, (0, 100))
        adjusted[k] = float(np.clip(adjusted[k], lo, hi))

    total = sum(adjusted.values())
    if total > 0:
        factor = 100.0 / total
        for k in adjusted:
            adjusted[k] = round(adjusted[k] * factor, 1)

    diff = round(100.0 - sum(adjusted.values()), 1)
    if diff != 0:
        biggest = max(adjusted, key=lambda k: adjusted[k])
        adjusted[biggest] = round(adjusted[biggest] + diff, 1)

    return adj_map, adjusted


# =============================================================================
# 6. FIX-5: KRİZ MOD KONTROLÜ
# =============================================================================
def check_crisis(eval_row: pd.Series) -> Tuple[bool, str]:
    """
    İki katmanlı kriz kontrolü:
      Soft: risk_score >= CRISIS_THRESHOLD (0.80)
      Hard: VIX > 40 veya CDS > 700 (anlık, gecikmesiz)

    Hard tetikleyici, risk_score'un toparlanma gecikmesini bypass eder.
    Döner: (is_crisis, kriz_neden)
    """
    risk  = float(eval_row.get("risk_score", 0.5) or 0.5)
    vix   = eval_row.get("vix_level", np.nan)
    cds   = eval_row.get("cds_level", np.nan)

    # Hard tetikleyiciler
    if pd.notna(vix) and float(vix) > VIX_CRISIS_LEVEL:
        return True, f"HARD-KRİZ: VIX={float(vix):.1f} > {VIX_CRISIS_LEVEL}"
    if pd.notna(cds) and float(cds) > CDS_CRISIS_LEVEL:
        return True, f"HARD-KRİZ: CDS={float(cds):.0f} > {CDS_CRISIS_LEVEL}"

    # Soft eşik
    if risk >= CRISIS_THRESHOLD:
        return True, f"SOFT-KRİZ: risk_score={risk:.3f} >= {CRISIS_THRESHOLD}"

    return False, ""


# =============================================================================
# 7. KARAR MOTORU — ANA FONKSİYON
# =============================================================================
def run_engine(profile: str,
               date: Optional[str] = None,
               amount: float = 100_000) -> dict:
    if profile not in BASE_PORTFOLIOS:
        raise ValueError(f"Geçersiz profil: {profile}. Seçenekler: {list(BASE_PORTFOLIOS)}")

    # 1. Veri
    df, eval_date = load_data(date)

    # 2. Derived
    df = compute_derived(df)
    eval_row = df.iloc[-1]

    # 3. FIX-5: Kriz kontrol
    is_crisis, crisis_reason = check_crisis(eval_row)
    risk_score = float(eval_row.get("risk_score", 0.5))

    # 4. Dağılım
    if is_crisis:
        final_alloc = CRISIS_OVERRIDE.copy()
        scores      = {a: 0.5 for a in ASSETS}
        adjustments = {a: 0.0 for a in ASSETS}
        mode        = "KRİZ"
    else:
        scores      = compute_asset_scores(df, eval_row)
        sensitivity = PROFILE_SENSITIVITY[profile]
        # FIX-7: apply_adjustments artık (adj_map, final_alloc) döndürüyor
        # adj_map base=0 varlıklara negatif adj uygulamaz → normalize bozulmuyor
        adjustments, final_alloc = apply_adjustments(BASE_PORTFOLIOS[profile], scores, sensitivity)
        mode        = "NORMAL"

    # 5. TL karşılıkları
    tl_alloc = {k: round(amount * v / 100, 2) for k, v in final_alloc.items()}

    # 6. Makro snapshot
    macro_cols = ["policy_rate", "real_rate", "usdtry", "usdtry_chg_30d",
                  "cpi_yoy", "bist_momentum", "fx_stress", "gold_real_return",
                  "vix_level", "cds_level", "risk_score"]
    macro_snapshot = {}
    for col in macro_cols:
        val = eval_row.get(col, np.nan)
        macro_snapshot[col] = round(float(val), 2) if pd.notna(val) else None

    # 7. Sinyaller
    signals = _generate_signals(macro_snapshot, scores, is_crisis, crisis_reason)

    return {
        "eval_date":    eval_date,
        "profile":      profile,
        "mode":         mode,
        "risk_score":   round(risk_score, 3),
        "is_crisis":    is_crisis,
        "crisis_reason": crisis_reason,
        "amount_tl":    amount,
        "base_alloc":   BASE_PORTFOLIOS[profile],
        "scores":       {k: round(v, 3) for k, v in scores.items()},
        "adjustments":  {k: round(v, 1) for k, v in adjustments.items()},
        "final_alloc":  final_alloc,
        "tl_alloc":     tl_alloc,
        "macro":        macro_snapshot,
        "signals":      signals,
        "is_backtest":  date is not None,
        "requested_date": date,
        "config": {
            "neutral_threshold": NEUTRAL_THRESHOLD,
            "rolling_window":    ROLLING_WINDOW,
            "crisis_threshold":  CRISIS_THRESHOLD,
            "vix_crisis_level":  VIX_CRISIS_LEVEL,
            "cds_crisis_level":  CDS_CRISIS_LEVEL,
            "asset_bounds":      ASSET_BOUNDS,
        },
    }


def _generate_signals(macro: dict, scores: dict,
                       is_crisis: bool, crisis_reason: str) -> list:
    signals = []

    if is_crisis:
        signals.append(f"KRIZ MODU: {crisis_reason}. Korumaci override aktif.")
        return signals

    rs  = macro.get("risk_score") or 0.5
    rr  = macro.get("real_rate")
    pr  = macro.get("policy_rate")
    fxs = macro.get("fx_stress")
    bm  = macro.get("bist_momentum")
    vix = macro.get("vix_level")
    cds = macro.get("cds_level")

    if rs > 0.65:   signals.append(f"Risk YUKSEK ({rs:.2f}): Korumaci varliklar one cekildi.")
    elif rs < 0.35: signals.append(f"Risk DUSUK ({rs:.2f}): Buyume varliklarina kapı acik.")
    else:           signals.append(f"Risk ORTA ({rs:.2f}): Dengeli yaklasim.")

    if rr is not None:
        if rr > 5:   signals.append(f"Reel faiz pozitif (%{rr:.1f}): Mevduat ve tahvil cazip.")
        elif rr < -5: signals.append(f"Reel faiz negatif (%{rr:.1f}): Enflasyon getiriyi eriyor.")

    if pr is not None:
        signals.append(f"Politika faizi: %{pr:.1f}")

    if fxs is not None and fxs > 0.05:
        signals.append(f"FX baskisi yuksek (stres={fxs:.3f}): Doviz koruması artırılıyor.")

    if bm is not None:
        if bm > 3:   signals.append(f"BIST momentum pozitif (%{bm:.1f}): Hisse cazibi artiyor.")
        elif bm < -3: signals.append(f"BIST momentum negatif (%{bm:.1f}): Hisse riski yuksek.")

    if vix is not None:
        lvl = "YUKSEK" if vix > 25 else "NORMAL"
        signals.append(f"VIX={vix:.1f} [{lvl}] (kriz esigi: >{VIX_CRISIS_LEVEL})")

    if cds is not None:
        lvl = "YUKSEK" if cds > 500 else "NORMAL"
        signals.append(f"CDS={cds:.0f} [{lvl}] (kriz esigi: >{CDS_CRISIS_LEVEL})")

    if scores:
        best  = max(scores, key=lambda k: scores[k])
        worst = min(scores, key=lambda k: scores[k])
        signals.append(f"En cazip varlik: {ASSET_LABELS[best]} (skor {scores[best]:.3f})")
        signals.append(f"En az cazip: {ASSET_LABELS[worst]} (skor {scores[worst]:.3f})")

    return signals


# =============================================================================
# 8. CLI
# =============================================================================
BAR_W = 28
PROFILE_LABELS = {
    "az_riskli":   "Az Riskli   (Koruyucu)",
    "orta_riskli": "Orta Riskli (Dengeli)",
    "cok_riskli":  "Cok Riskli  (Agresif)",
}


def _bar(pct: float) -> str:
    n = int(pct / 100 * BAR_W)
    return "\u2588" * n + "\u2591" * (BAR_W - n)


def print_result(result: dict) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  PORTFOY KARAR MOTORU — {result['eval_date']}  [v6]")
    print(sep)
    print(f"  Profil      : {PROFILE_LABELS.get(result['profile'], result['profile'])}")
    print(f"  Mod         : {result['mode']}"
          + (f"  [{result['crisis_reason']}]" if result["is_crisis"] else ""))
    print(f"  Risk skoru  : {result['risk_score']:.3f}")
    print(f"  Tutar       : {result['amount_tl']:,.0f} TL")
    cfg = result["config"]
    print(f"  [Ayarlar]   neutral_thr={cfg['neutral_threshold']}  "
          f"rolling={cfg['rolling_window']}g  "
          f"VIX_kriz>{cfg['vix_crisis_level']}  CDS_kriz>{cfg['cds_crisis_level']}")
    print(sep)

    m = result["macro"]
    print("\n  MAKRO")
    print(f"  {'Politika Faizi':<24}: %{m.get('policy_rate','N/A')}")
    print(f"  {'Reel Faiz':<24}: %{m.get('real_rate','N/A')}")
    print(f"  {'CPI (yillik)':<24}: %{m.get('cpi_yoy','N/A')}")
    print(f"  {'USD/TRY 30g degisim':<24}: %{m.get('usdtry_chg_30d','N/A')}")
    print(f"  {'BIST Momentum':<24}: %{m.get('bist_momentum','N/A')}")
    print(f"  {'VIX':<24}: {m.get('vix_level','N/A')}")
    print(f"  {'CDS 5Y':<24}: {m.get('cds_level','N/A')}")

    print(f"\n  {'─'*63}")
    print(f"  {'Varlik':<22} {'Base%':>5} {'Skor':>6} {'Ayar':>6} {'Final%':>7}  {'TL':>12}  Bar")
    print(f"  {'─'*63}")

    for asset in ASSETS:
        label = ASSET_LABELS[asset]
        b     = result["base_alloc"].get(asset, 0)
        sc    = result["scores"].get(asset, 0.5)
        a     = result["adjustments"].get(asset, 0)
        f     = result["final_alloc"].get(asset, 0)
        t     = result["tl_alloc"].get(asset, 0)
        a_str = f"+{a:.0f}" if a > 0 else f"{a:.0f}" if a < 0 else "  0"
        print(f"  {label:<22} {b:>4.0f}% {sc:>5.3f} {a_str:>5} {f:>6.1f}%  {t:>11,.0f}  {_bar(f)}")

    print(f"  {'─'*63}")
    total_tl = sum(result["tl_alloc"].values())
    print(f"  {'TOPLAM':<22}  100%              100.0%  {total_tl:>11,.0f}")

    print(f"\n  SINYALLER")
    for sig in result["signals"]:
        print(f"  {sig}")
    print(f"\n{sep}\n")


def save_result(result: dict) -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ENGINE_OUT / f"decision_{result['profile']}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Kaydedildi: {path}")


def interactive_prompt() -> Tuple[str, float]:
    print("\n" + "=" * 52)
    print("  PORTFOY KARAR MOTORU  v6")
    print("=" * 52)
    print("  1  Az Riskli    (Koruyucu)")
    print("  2  Orta Riskli  (Dengeli)")
    print("  3  Cok Riskli   (Agresif)")
    print("=" * 52)
    while True:
        c = input("  Secim (1/2/3): ").strip()
        if c == "1":   profile = "az_riskli";   break
        elif c == "2": profile = "orta_riskli"; break
        elif c == "3": profile = "cok_riskli";  break
        else: print("  1, 2 veya 3 girin.")
    while True:
        s = input("  Portfoy (TL, varsayilan 100000): ").strip()
        if not s: return profile, 100_000.0
        try:
            a = float(s.replace(",", "").replace(".", ""))
            if a > 0: return profile, a
        except ValueError: pass
        print("  Gecersiz tutar.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(BASE_PORTFOLIOS), default=None)
    parser.add_argument("--date",    default=None)
    parser.add_argument("--amount",  type=float, default=100_000)
    parser.add_argument("--save",    action="store_true")
    args = parser.parse_args()

    try:
        if args.profile:
            profile, amount = args.profile, args.amount
        else:
            profile, amount = interactive_prompt()
        print(f"\n  Hesaplaniyor... ({profile})")
        result = run_engine(profile=profile, date=args.date, amount=amount)
        print_result(result)
        if args.save:
            save_result(result)
    except FileNotFoundError as e:
        print(f"\n[HATA] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[HATA] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# BACKTEST & CLI EXTENSIONS (v6.1)
# =============================================================================

def get_available_dates() -> list:
    """Pipeline verisinde mevcut tarihleri döner."""
    if not MERGED_CSV.exists():
        return []
    df = pd.read_csv(MERGED_CSV, parse_dates=["date"], usecols=["date"])
    return df["date"].dt.strftime("%Y-%m-%d").tolist()


def run_backtest(profile: str, dates: list, amount: float = 100_000) -> list:
    """
    Birden fazla tarih için motor çalıştırır.
    Kullanım: karşılaştırmalı analiz ve trend görselleştirme.
    """
    results = []
    for d in dates:
        try:
            r = run_engine(profile=profile, date=d, amount=amount)
            results.append(r)
        except Exception as e:
            results.append({"eval_date": d, "error": str(e), "profile": profile})
    return results


def print_backtest_comparison(results: list) -> None:
    """Backtest sonuçlarını yan yana karşılaştırma tablosu."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("Backtest sonucu yok.")
        return
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  BACKTEST: {valid[0]['profile'].upper()}  —  {len(valid)} tarih")
    print(sep)

    # Header
    h = f"  {'Varlik':<22}"
    for r in valid:
        h += f"  {r['eval_date']:>12}"
    print(h)
    print(f"  {'─'*min(90,22+14*len(valid))}")

    for asset in ASSETS:
        row = f"  {ASSET_LABELS[asset]:<22}"
        for r in valid:
            f_val = r["final_alloc"].get(asset, 0)
            row += f"  {f_val:>11.1f}%"
        print(row)

    print(f"  {'─'*min(90,22+14*len(valid))}")
    risk_row = f"  {'Risk Skoru':<22}"
    mode_row = f"  {'Mod':<22}"
    for r in valid:
        risk_row += f"  {r['risk_score']:>12.3f}"
        mode_row += f"  {r['mode']:>12}"
    print(risk_row)
    print(mode_row)
    print(f"\n{sep}\n")