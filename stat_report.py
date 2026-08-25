from __future__ import annotations
"""
stat_report.py  —  Karar Motorunun İstatistiksel Temellerini Raporlar (PDF)

Kullanım:
  python stat_report.py                        # orta_riskli, PDF çıktı
  python stat_report.py --profile az_riskli
  python stat_report.py --profile cok_riskli --output rapor.pdf

Python 3.10 uyumlu (f-string içinde backslash yok).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decision_engine import (
    BASE_PORTFOLIOS, ASSETS, ASSET_LABELS,
    PROFILE_SENSITIVITY, CRISIS_THRESHOLD, ADJ,
    load_data, compute_derived, compute_asset_scores,
    score_to_adjustment, apply_adjustments, percentile_rank,
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Renkler ──────────────────────────────────────────────────────────────────
C_BG       = colors.HexColor("#0F0F1A")
C_SURFACE  = colors.HexColor("#16162A")
C_ACCENT   = colors.HexColor("#7C6AF7")
C_ACCENT2  = colors.HexColor("#A78BFA")
C_GREEN    = colors.HexColor("#34D399")
C_RED      = colors.HexColor("#F87171")
C_AMBER    = colors.HexColor("#FBBF24")
C_TEAL     = colors.HexColor("#2DD4BF")
C_TEXT     = colors.HexColor("#E8E8F0")
C_MUTED    = colors.HexColor("#6B6B8A")
C_BORDER   = colors.HexColor("#2A2A3E")
C_WHITE    = colors.white
C_BLACK    = colors.black

# ── Stiller ───────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title = ps("RptTitle",
        fontName="Helvetica-Bold", fontSize=20,
        textColor=C_ACCENT2, alignment=TA_LEFT,
        spaceAfter=4, leading=24)

    subtitle = ps("RptSubtitle",
        fontName="Helvetica", fontSize=10,
        textColor=C_MUTED, alignment=TA_LEFT,
        spaceAfter=14, leading=14)

    section = ps("RptSection",
        fontName="Helvetica-Bold", fontSize=12,
        textColor=C_ACCENT, alignment=TA_LEFT,
        spaceBefore=18, spaceAfter=6, leading=16)

    body = ps("RptBody",
        fontName="Helvetica", fontSize=9,
        textColor=C_TEXT, alignment=TA_LEFT,
        spaceAfter=4, leading=14)

    mono = ps("RptMono",
        fontName="Courier", fontSize=8,
        textColor=C_TEXT, alignment=TA_LEFT,
        spaceAfter=3, leading=12)

    mono_muted = ps("RptMonoMuted",
        fontName="Courier", fontSize=8,
        textColor=C_MUTED, alignment=TA_LEFT,
        spaceAfter=2, leading=12)

    label = ps("RptLabel",
        fontName="Helvetica-Bold", fontSize=8,
        textColor=C_MUTED, alignment=TA_LEFT,
        spaceAfter=2, leading=11, spaceBefore=8)

    caption = ps("RptCaption",
        fontName="Helvetica-Oblique", fontSize=8,
        textColor=C_MUTED, alignment=TA_LEFT,
        spaceAfter=6, leading=12)

    th_c = ps("ThC",
        fontName="Helvetica-Bold", fontSize=8,
        textColor=C_WHITE, alignment=TA_CENTER, leading=11)

    th_l = ps("ThL",
        fontName="Helvetica-Bold", fontSize=8,
        textColor=C_WHITE, alignment=TA_LEFT, leading=11)

    th_r = ps("ThR",
        fontName="Helvetica-Bold", fontSize=8,
        textColor=C_WHITE, alignment=TA_RIGHT, leading=11)

    td_c = ps("TdC",
        fontName="Courier", fontSize=8,
        textColor=C_TEXT, alignment=TA_CENTER, leading=11)

    td_l = ps("TdL",
        fontName="Courier", fontSize=8,
        textColor=C_TEXT, alignment=TA_LEFT, leading=11)

    td_r = ps("TdR",
        fontName="Courier", fontSize=8,
        textColor=C_TEXT, alignment=TA_RIGHT, leading=11)

    td_green = ps("TdGreen",
        fontName="Courier-Bold", fontSize=8,
        textColor=C_GREEN, alignment=TA_RIGHT, leading=11)

    td_red = ps("TdRed",
        fontName="Courier-Bold", fontSize=8,
        textColor=C_RED, alignment=TA_RIGHT, leading=11)

    td_amber = ps("TdAmber",
        fontName="Courier-Bold", fontSize=8,
        textColor=C_AMBER, alignment=TA_RIGHT, leading=11)

    return {
        "RptTitle": title, "RptSubtitle": subtitle, "RptSection": section,
        "RptBody": body, "RptMono": mono, "RptMonoMuted": mono_muted,
        "label": label, "RptCaption": caption,
        "ThC": th_c, "ThL": th_l, "ThR": th_r,
        "TdC": td_c, "TdL": td_l, "TdR": td_r,
        "TdGreen": td_green, "TdRed": td_red, "TdAmber": td_amber,
    }


def bar_str(v: float, width: int = 18) -> str:
    filled = max(0, min(width, int(v * width)))
    return "\u2588" * filled + "\u2591" * (width - filled)

def score_color(s: float):
    if s >= 0.65: return C_GREEN
    if s >= 0.45: return C_AMBER
    return C_RED

def score_label(s: float) -> str:
    if s >= 0.65: return "YUKSEK CAZIP"
    if s >= 0.55: return "cazip"
    if s >= 0.45: return "notr"
    if s >= 0.35: return "az cazip"
    return "DUSUK CAZIP"

def risk_label(r: float) -> str:
    if r >= CRISIS_THRESHOLD: return f"KRIZ ({r:.3f})"
    if r >= 0.65: return f"YUKSEK ({r:.3f})"
    if r >= 0.35: return f"ORTA ({r:.3f})"
    return f"DUSUK ({r:.3f})"

def adj_color(a: float):
    if a > 0.5: return C_GREEN
    if a < -0.5: return C_RED
    return C_MUTED

def fmt(v, dec=2, sfx=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{dec}f}{sfx}"

# ── Tablo yardımcıları ────────────────────────────────────────────────────────
BASE_TS = TableStyle([
    ("BACKGROUND",   (0, 0), (-1, 0),  C_ACCENT),
    ("TEXTCOLOR",    (0, 0), (-1, 0),  C_WHITE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_SURFACE, C_BG]),
    ("GRID",         (0, 0), (-1, -1), 0.3, C_BORDER),
    ("TOPPADDING",   (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
])

def make_table(data, col_widths, extra_style=None):
    ts = TableStyle(BASE_TS.getCommands())
    if extra_style:
        for cmd in extra_style:
            ts.add(*cmd)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    return t

# ── RAPOR ─────────────────────────────────────────────────────────────────────
def build_pdf(profile: str, output_path: str) -> None:
    S = make_styles()
    PAGE_W, PAGE_H = A4
    margin = 1.8 * cm
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=f"Portfoy Karar Motoru - Istatistik Raporu ({profile})",
        author="Portfolio Decision Engine v5",
    )

    # Veri hazırla
    df, eval_date = load_data()
    df = compute_derived(df)
    row = df.iloc[-1]
    base        = BASE_PORTFOLIOS[profile]
    sensitivity = PROFILE_SENSITIVITY[profile]
    scores      = compute_asset_scores(df, row)
    adjustments, final_alloc = apply_adjustments(base, scores, sensitivity)
    risk_score  = float(row.get("risk_score", 0.5))
    is_crisis   = risk_score >= CRISIS_THRESHOLD

    content = []
    W = PAGE_W - 2 * margin   # kullanılabilir genişlik ~157 mm

    def add(*items):
        content.extend(items)

    def hr(color=C_BORDER, thickness=0.5):
        return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)

    def sp(h=6):
        return Spacer(1, h)

    # ── KAPAK BAŞLIĞI ──
    add(
        Paragraph("PORTFOY KARAR MOTORU", S["RptTitle"]),
        Paragraph(f"Istatistiksel Seffaflik Raporu  |  Profil: {profile.upper().replace('_',' ')}  |  Tarih: {eval_date}", S["RptSubtitle"]),
        hr(C_ACCENT, 1),
        sp(4),
    )

    # Durum kartı (tek satır tablo)
    mode_str = "KRIZ — override aktif" if is_crisis else "NORMAL — dinamik ayarlama"
    risk_c   = C_RED if risk_score >= 0.65 else C_AMBER if risk_score >= 0.35 else C_GREEN
    status_data = [
        [Paragraph("MOD", S["ThC"]),    Paragraph("RISK SKORU", S["ThC"]),
         Paragraph("PROFIL", S["ThC"]), Paragraph("KRIZ ESIGI", S["ThC"])],
        [Paragraph(mode_str, S["TdC"]), Paragraph(f"{risk_score:.4f}", S["TdC"]),
         Paragraph(profile.replace("_"," "), S["TdC"]),
         Paragraph(str(CRISIS_THRESHOLD), S["TdC"])],
    ]
    status_ts = TableStyle(BASE_TS.getCommands())
    status_ts.add("TEXTCOLOR", (1, 1), (1, 1), risk_c)
    status_ts.add("FONTNAME",  (1, 1), (1, 1), "Courier-Bold")
    st = Table(status_data, colWidths=[W*0.28, W*0.20, W*0.27, W*0.25])
    st.setStyle(status_ts)
    add(st, sp(14))

    # ══ BÖLÜM 1 — TÜREYEN DEĞİŞKENLER ══
    add(
        Paragraph("BOLUM 1: TUREYEN (DERIVED) DEGISKENLER", S["RptSection"]),
        hr(),
        Paragraph(
            "Ham veriden hesaplanan sinyal degiskenleri. Motor ham veriyi degil, "
            "bu turetilmis degiskenleri okur. Asagida her degiskenin formulu, "
            "guncel degeri ve yorumu verilmistir.",
            S["RptBody"]),
        sp(6),
    )

    derived_rows = [
        [Paragraph("Degisken", S["ThL"]), Paragraph("Formul", S["ThL"]),
         Paragraph("Deger", S["ThC"]),    Paragraph("Yorum", S["ThL"])],
    ]
    derived_defs = [
        ("real_rate",        "Reel Faiz (%)",
         "policy_rate - cpi_yoy",
         "Pozitif: mevduat/tahvil cazip. Negatif: enflasyon getiriyi eriyor."),
        ("cpi_yoy",          "CPI Yillik (%)",
         "cpi_index.pct_change(252) x 100",
         "Yillik enflasyon. Yuksekse reel faiz negatife duser."),
        ("usdtry_chg_30d",   "USD/TRY 30g Degisim (%)",
         "usdtry.pct_change(30) x 100",
         "Doviz baskisinin 30 gunluk birikimi."),
        ("fx_stress",        "FX Stres (0-1)",
         "std(usdtry,30) / mean(usdtry,30)",
         "0: kur sakin. 1: cok dalgali. Doviz/altin kararini tetikler."),
        ("bist_momentum",    "BIST Momentum (%)",
         "(bist100 - MA20) / MA20 x 100",
         "Pozitif: borsada yukselis trendi. Hisse kararini tetikler."),
        ("gold_real_return", "Altin Reel Getiri (%)",
         "gold_try.pct_change(252) x 100 - cpi_yoy",
         "Altinin TRY enflasyonunu yenip yenmedigini olcer."),
        ("risk_score",       "Bilesik Risk Skoru (0-1)",
         "agirlikli: vix*0.20+fx*0.25+rr*0.20+cds*0.20+bist*0.15",
         f"0=dusuk risk, 1=yuksek risk. Kriz esigi: {CRISIS_THRESHOLD}"),
    ]

    for col, label, formula, yorum in derived_defs:
        val     = row.get(col, np.nan)
        val_str = fmt(val, 2)
        derived_rows.append([
            Paragraph(label,   S["TdL"]),
            Paragraph(formula, S["RptMono"]),
            Paragraph(val_str, S["TdC"]),
            Paragraph(yorum,   S["TdL"]),
        ])

    dt = make_table(derived_rows, [W*0.20, W*0.30, W*0.10, W*0.40])
    add(dt, sp(10))

    # Risk skoru bileşen tablosu
    add(Paragraph("Risk Skoru Bilesenleri", S["label"]), sp(2))
    rc_rows = [
        [Paragraph("Bilesen", S["ThL"]), Paragraph("Deger", S["ThC"]),
         Paragraph("Norm (0-1)", S["ThC"]), Paragraph("Agirlik", S["ThC"]),
         Paragraph("Katki", S["ThC"]), Paragraph("Bar", S["ThL"])],
    ]
    risk_components = {
        "vix":       (row.get("vix_level",   np.nan), 0.20,  10, 50,  False),
        "fx_stress": (row.get("fx_stress",   np.nan), 0.25,   0,  1,  False),
        "real_rate": (row.get("real_rate",   np.nan), 0.20, -50, 50,  True),
        "cds":       (row.get("cds_level",   np.nan), 0.20, 200, 800, False),
        "bist":      (row.get("bist_momentum",np.nan),0.15, -30, 30,  True),
    }
    total_risk = 0.0
    for name, (val, w, lo, hi, invert) in risk_components.items():
        if pd.isna(val):
            continue
        if invert:
            norm = (-float(val) - (-abs(lo))) / (abs(hi) + abs(lo)) if lo < 0 else (-float(val) + hi) / (2 * hi)
            if name == "real_rate":
                norm = (-float(val) + 50) / 100
            elif name == "bist":
                norm = (-float(val) + 30) / 60
        else:
            norm = (float(val) - lo) / (hi - lo)
        norm  = float(np.clip(norm, 0, 1))
        katki = norm * w
        total_risk += katki
        bar = bar_str(norm, 12)
        rc_rows.append([
            Paragraph(name,           S["TdL"]),
            Paragraph(fmt(val, 2),    S["TdC"]),
            Paragraph(fmt(norm, 3),   S["TdC"]),
            Paragraph(f"{w:.2f}",     S["TdC"]),
            Paragraph(fmt(katki, 4),  S["TdC"]),
            Paragraph(bar,            S["RptMono"]),
        ])

    rc_ts_extra = [
        ("BACKGROUND", (0, -1), (-1, -1), C_BORDER),
    ]
    rc_rows.append([
        Paragraph("TOPLAM", S["ThL"]),
        Paragraph("", S["TdC"]),
        Paragraph("", S["TdC"]),
        Paragraph("", S["TdC"]),
        Paragraph(fmt(total_risk, 4), S["TdC"]),
        Paragraph(risk_label(risk_score), S["TdL"]),
    ])
    add(make_table(rc_rows, [W*0.14, W*0.12, W*0.14, W*0.12, W*0.12, W*0.36], rc_ts_extra), sp(10))

    # ══ BÖLÜM 2 — PERCENTİLE SCORING ══
    add(
        Paragraph("BOLUM 2: PERCENTILE BAZLI VARLIK SKORLARI", S["RptSection"]),
        hr(),
        Paragraph(
            "Her varlik icin 'su an ne kadar cazip?' skoru (0-1). "
            "Yontem: sinyal degiskeninin historik dagilimindaki yuzdedilimi hesapla, "
            "sonra agirlikli birlestir. 0=hic cazip degil, 0.5=notr, 1=cok cazip.",
            S["RptBody"]),
        sp(6),
    )

    # Sinyal percentile tablosu
    add(Paragraph("Sinyal Degiskenlerinin Historik Pozisyonlari", S["label"]), sp(2))
    sp_rows = [
        [Paragraph("Degisken", S["ThL"]), Paragraph("Guncel Deger", S["ThC"]),
         Paragraph("Percentile", S["ThC"]), Paragraph("Trend", S["ThC"]),
         Paragraph("Aciklama", S["ThL"])],
    ]
    sinyal_cols = [
        ("real_rate",        "Reel Faiz (%)",     True,  "yuksek=iyi (mevduat cazip)"),
        ("fx_stress",        "FX Stres",          False, "yuksek=kotu (kur baskisi)"),
        ("bist_momentum",    "BIST Momentum (%)", True,  "yuksek=iyi (borsa guclu)"),
        ("gold_real_return", "Altin Reel Getiri", True,  "yuksek=iyi (altin kazandiriyor)"),
        ("us10y",            "US 10Y Getiri (%)", False, "yuksek=kotu (global faiz yukari)"),
        ("vix_level",        "VIX",               False, "yuksek=kotu (global panik)"),
        ("risk_score",       "Risk Skoru",        False, "yuksek=kotu (genel risk)"),
    ]
    for col, label, higher_good, aciklama in sinyal_cols:
        if col not in df.columns or df[col].dropna().empty:
            continue
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        pct     = percentile_rank(df[col], float(val))
        adj_pct = (1 - pct) if not higher_good else pct
        if adj_pct > 0.6:
            trend, tc = "guclu yukari", C_GREEN
        elif adj_pct < 0.4:
            trend, tc = "zayif asagi", C_RED
        else:
            trend, tc = "notr", C_AMBER

        trend_style = ParagraphStyle("TrendTmp", parent=S["TdC"], textColor=tc)
        sp_rows.append([
            Paragraph(label,               S["TdL"]),
            Paragraph(fmt(val, 2),         S["TdC"]),
            Paragraph(f"%{pct*100:.1f}",   S["TdC"]),
            Paragraph(trend,               trend_style),
            Paragraph(aciklama,            S["TdL"]),
        ])
    add(make_table(sp_rows, [W*0.22, W*0.14, W*0.12, W*0.16, W*0.36]), sp(10))

    # Varlık skor formülleri tablosu
    add(Paragraph("Varlik Skor Formulleri ve Hesaplari", S["label"]), sp(2))

    def safe_pct(col, reverse=False):
        if col not in df.columns or df[col].dropna().empty:
            return 0.5
        v = row.get(col, np.nan)
        if pd.isna(v): return 0.5
        r = percentile_rank(df[col], float(v))
        return 1 - r if reverse else r

    rr_pct   = safe_pct("real_rate")
    bist_pct = safe_pct("bist_momentum")
    grr_pct  = safe_pct("gold_real_return")
    fxs_pct  = safe_pct("fx_stress")
    us10_inv = safe_pct("us10y", reverse=True)
    risk     = float(row.get("risk_score", 0.5))

    score_defs = {
        "mevduat":       (f"rr_pct*0.6 + risk*0.4",
                          f"{rr_pct:.3f}*0.6 + {risk:.3f}*0.4",
                          f"{rr_pct*0.6:.4f} + {risk*0.4:.4f}"),
        "doviz":         (f"fx_pct*0.5 + risk*0.3 + (1-rr)*0.2",
                          f"{fxs_pct:.3f}*0.5 + {risk:.3f}*0.3 + {1-rr_pct:.3f}*0.2",
                          f"{fxs_pct*0.5:.4f}+{risk*0.3:.4f}+{(1-rr_pct)*0.2:.4f}"),
        "altin":         (f"risk*0.5 + grr_pct*0.3 + fx_pct*0.2",
                          f"{risk:.3f}*0.5 + {grr_pct:.3f}*0.3 + {fxs_pct:.3f}*0.2",
                          f"{risk*0.5:.4f}+{grr_pct*0.3:.4f}+{fxs_pct*0.2:.4f}"),
        "tahvil":        (f"rr_pct*0.5 + (1-risk)*0.3 + us10_inv*0.2",
                          f"{rr_pct:.3f}*0.5 + {1-risk:.3f}*0.3 + {us10_inv:.3f}*0.2",
                          f"{rr_pct*0.5:.4f}+{(1-risk)*0.3:.4f}+{us10_inv*0.2:.4f}"),
        "yatirim_fonu":  (f"(1-|bist-0.5|*2)*0.4 + (1-risk)*0.3 + rr*0.3",
                          f"{(1-abs(bist_pct-0.5)*2):.3f}*0.4+{1-risk:.3f}*0.3+{rr_pct:.3f}*0.3",
                          f"{(1-abs(bist_pct-0.5)*2)*0.4:.4f}+{(1-risk)*0.3:.4f}+{rr_pct*0.3:.4f}"),
        "hisse":         (f"bist_pct*0.5 + (1-risk)*0.5",
                          f"{bist_pct:.3f}*0.5 + {1-risk:.3f}*0.5",
                          f"{bist_pct*0.5:.4f}+{(1-risk)*0.5:.4f}"),
        "temettu_hisse": (f"(1-rr)*0.4 + (1-risk)*0.4 + bist*0.2",
                          f"{1-rr_pct:.3f}*0.4+{1-risk:.3f}*0.4+{bist_pct:.3f}*0.2",
                          f"{(1-rr_pct)*0.4:.4f}+{(1-risk)*0.4:.4f}+{bist_pct*0.2:.4f}"),
        "kripto":        (f"(1-risk)*0.6 + us10_inv*0.4",
                          f"{1-risk:.3f}*0.6 + {us10_inv:.3f}*0.4",
                          f"{(1-risk)*0.6:.4f}+{us10_inv*0.4:.4f}"),
    }

    sc_rows = [
        [Paragraph("Varlik",  S["ThL"]), Paragraph("Formul",  S["ThL"]),
         Paragraph("Hesap",   S["ThL"]), Paragraph("Skor",    S["ThC"]),
         Paragraph("Bar",     S["ThL"]), Paragraph("Durum",   S["ThL"])],
    ]
    for asset in ASSETS:
        sc   = scores[asset]
        f1, f2, f3 = score_defs[asset]
        sc_style = ParagraphStyle("ScTmp", parent=S["TdC"], textColor=score_color(sc))
        sc_rows.append([
            Paragraph(ASSET_LABELS[asset], S["TdL"]),
            Paragraph(f1,                  S["RptMono"]),
            Paragraph(f3,                  S["RptMono"]),
            Paragraph(f"{sc:.4f}",         sc_style),
            Paragraph(bar_str(sc, 10),     S["RptMono"]),
            Paragraph(score_label(sc),     S["TdL"]),
        ])
    add(make_table(sc_rows, [W*0.16, W*0.22, W*0.22, W*0.08, W*0.12, W*0.20]), sp(10))

    # ══ BÖLÜM 3 — AYARLAMA ══
    add(
        Paragraph("BOLUM 3: AYARLAMA MOTORU  (skor + profile -> +/-pt)", S["RptSection"]),
        hr(),
        Paragraph(
            f"Profil sensitivitesi: {sensitivity}  ({profile}). "
            "Ayni skor, farkli profillerde farkli buyuklukte ayarlama uretiyor. "
            "Sapma = skor - 0.5. Buyuk sapma = buyuk ayar.",
            S["RptBody"]),
        sp(6),
    )

    # Eşik tablosu
    esik_rows = [
        [Paragraph("|Sapma| Araligi", S["ThL"]),
         Paragraph("Ham Ayar (pt)", S["ThC"]),
         Paragraph(f"x Profil ({sensitivity})", S["ThC"]),
         Paragraph("Net Ayar (pt)", S["ThC"])],
        [Paragraph("|dev| < 0.15",        S["TdL"]), Paragraph("0",              S["TdC"]),
         Paragraph(f"x {sensitivity}",    S["TdC"]), Paragraph("0",              S["TdC"])],
        [Paragraph("0.15 <= |dev| < 0.30",S["TdL"]), Paragraph(f"+/-{ADJ['hafif']}",S["TdC"]),
         Paragraph(f"x {sensitivity}",    S["TdC"]), Paragraph(f"+/-{ADJ['hafif']*sensitivity:.1f}", S["TdC"])],
        [Paragraph("0.30 <= |dev| < 0.45",S["TdL"]), Paragraph(f"+/-{ADJ['orta']}", S["TdC"]),
         Paragraph(f"x {sensitivity}",    S["TdC"]), Paragraph(f"+/-{ADJ['orta']*sensitivity:.1f}",  S["TdC"])],
        [Paragraph("|dev| >= 0.45",       S["TdL"]), Paragraph(f"+/-{ADJ['guclu']}",S["TdC"]),
         Paragraph(f"x {sensitivity}",    S["TdC"]), Paragraph(f"+/-{ADJ['guclu']*sensitivity:.1f}",  S["TdC"])],
    ]
    add(make_table(esik_rows, [W*0.35, W*0.20, W*0.25, W*0.20]), sp(8))

    # Varlık ayarlama tablosu
    adj_rows = [
        [Paragraph("Varlik",    S["ThL"]), Paragraph("Skor",   S["ThC"]),
         Paragraph("Sapma",     S["ThC"]), Paragraph("|Sapma|",S["ThC"]),
         Paragraph("Bolge",     S["ThC"]), Paragraph("Ayar pt",S["ThC"])],
    ]
    for asset in ASSETS:
        sc     = scores[asset]
        dev    = sc - 0.5
        abs_d  = abs(dev)
        if   abs_d < 0.15: bolge, raw = "NOTR",  0
        elif abs_d < 0.30: bolge, raw = "hafif",  ADJ["hafif"]
        elif abs_d < 0.45: bolge, raw = "orta",   ADJ["orta"]
        else:              bolge, raw = "GUCLU",  ADJ["guclu"]
        net    = (1 if dev > 0 else -1) * raw * sensitivity
        nc_    = C_GREEN if net > 0.5 else C_RED if net < -0.5 else C_MUTED
        ns     = f"+{net:.1f}" if net > 0 else f"{net:.1f}" if net < 0 else "0"
        ns_sty = ParagraphStyle("NSTmp", parent=S["TdC"], textColor=nc_)
        adj_rows.append([
            Paragraph(ASSET_LABELS[asset], S["TdL"]),
            Paragraph(f"{sc:.3f}",         S["TdC"]),
            Paragraph(f"{dev:+.3f}",       S["TdC"]),
            Paragraph(f"{abs_d:.3f}",      S["TdC"]),
            Paragraph(bolge,               S["TdC"]),
            Paragraph(ns,                  ns_sty),
        ])
    add(make_table(adj_rows, [W*0.28, W*0.12, W*0.13, W*0.13, W*0.17, W*0.17]), sp(10))

    # ══ BÖLÜM 4 — NORMALİZASYON ══
    add(
        Paragraph("BOLUM 4: NORMALIZASYON  (toplam = %100)", S["RptSection"]),
        hr(),
        Paragraph(
            "Ayarlamalar sonrasi negatife dusen varliklar 0'a kilitlenir. "
            "Ardindan tum pozisyonlar toplamin 100'e esit olacagi sekilde olceklenir.",
            S["RptBody"]),
        sp(6),
    )

    norm_rows = [
        [Paragraph("Varlik",      S["ThL"]), Paragraph("Base%", S["ThC"]),
         Paragraph("Ayar",        S["ThC"]), Paragraph("Ham",   S["ThC"]),
         Paragraph("Klip",        S["ThC"]), Paragraph("Final%",S["ThC"])],
    ]
    clip_dict = {}
    for asset in ASSETS:
        b    = base.get(asset, 0)
        a    = adjustments.get(asset, 0)
        raw  = b + a
        clip = max(0, raw)
        clip_dict[asset] = clip

    raw_sum = sum(clip_dict.values())
    for asset in ASSETS:
        b    = base.get(asset, 0)
        a    = adjustments.get(asset, 0)
        raw  = b + a
        clip = clip_dict[asset]
        fin  = final_alloc.get(asset, 0)
        a_s  = f"+{a:.1f}" if a > 0 else f"{a:.1f}" if a < 0 else "0"
        a_c  = C_GREEN if a > 0.5 else C_RED if a < -0.5 else C_MUTED
        klip = "->0" if raw < 0 else ""
        a_sty = ParagraphStyle("ATmp", parent=S["TdC"], textColor=a_c)
        norm_rows.append([
            Paragraph(ASSET_LABELS[asset], S["TdL"]),
            Paragraph(f"{b:.1f}%",         S["TdC"]),
            Paragraph(a_s,                 a_sty),
            Paragraph(f"{raw:.1f}",        S["TdC"]),
            Paragraph(klip,                S["TdC"]),
            Paragraph(f"{fin:.1f}%",       S["TdC"]),
        ])

    scale_str = f"{100/raw_sum:.4f}" if raw_sum > 0 else "N/A"
    norm_rows.append([
        Paragraph("Olcek faktoru", S["ThL"]),
        Paragraph("",  S["TdC"]),
        Paragraph("",  S["TdC"]),
        Paragraph(f"Klip sonrasi toplam: {raw_sum:.1f}", S["TdL"]),
        Paragraph("",  S["TdC"]),
        Paragraph(f"100/{raw_sum:.1f} = {scale_str}", S["TdC"]),
    ])
    add(make_table(norm_rows, [W*0.28, W*0.10, W*0.10, W*0.15, W*0.10, W*0.27]), sp(10))

    # ══ BÖLÜM 5 — KARAR ÖZETİ ══
    add(
        Paragraph("BOLUM 5: KARAR OZETI", S["RptSection"]),
        hr(),
    )

    final_rows = [
        [Paragraph("Varlik",      S["ThL"]), Paragraph("Base%", S["ThC"]),
         Paragraph("Ayar pt",     S["ThC"]), Paragraph("Final%",S["ThC"]),
         Paragraph("Degisim",     S["ThC"]), Paragraph("Bar",   S["ThL"])],
    ]
    for asset in ASSETS:
        b    = base.get(asset, 0)
        a    = adjustments.get(asset, 0)
        fin  = final_alloc.get(asset, 0)
        diff = fin - b
        a_s  = f"+{a:.1f}" if a > 0 else f"{a:.1f}" if a < 0 else "0"
        d_s  = f"+{diff:.1f}pt" if diff > 0.5 else f"{diff:.1f}pt" if diff < -0.5 else "degismedi"
        d_c  = C_GREEN if diff > 0.5 else C_RED if diff < -0.5 else C_MUTED
        a_c  = C_GREEN if a > 0.5 else C_RED if a < -0.5 else C_MUTED
        a_sty = ParagraphStyle("ASTmp", parent=S["TdC"], textColor=a_c)
        d_sty = ParagraphStyle("DSTmp", parent=S["TdC"], textColor=d_c)
        bar   = bar_str(fin / 100, 15)
        final_rows.append([
            Paragraph(ASSET_LABELS[asset], S["TdL"]),
            Paragraph(f"{b:.1f}%",         S["TdC"]),
            Paragraph(a_s,                 a_sty),
            Paragraph(f"{fin:.1f}%",       S["TdC"]),
            Paragraph(d_s,                 d_sty),
            Paragraph(bar,                 S["RptMono"]),
        ])
    add(make_table(final_rows, [W*0.25, W*0.10, W*0.10, W*0.10, W*0.18, W*0.27]), sp(10))

    # ══ BÖLÜM 6 — GELİŞTİRME ══
    add(
        Paragraph("BOLUM 6: GELISTIRME YOLLARI", S["RptSection"]),
        hr(),
        Paragraph(
            "Mevcut sistemin sinirlamalari ve onerileri. "
            "Her madde: SORUN — ONERI seklinde yapilandi.",
            S["RptBody"]),
        sp(6),
    )

    gelistirme = [
        ("SKOR AGIRLIKLARI",
         "Mevcut agirliklar (0.4, 0.5, 0.6 vb.) sezgisel belirlendi. "
         "Oneri: Gecmis verilerle backtest yaparak optimize et. "
         "minimize(portfoy_volatilitesi) veya maximize(sharpe_ratio)."),
        ("PERCENTILE PENCERESI",
         "Su an tum historik veri tek pencere kullaniliyor. "
         "Erken donem veriler guncel rejimi bozuyor. "
         "Oneri: rolling 252-gunluk (1 yillik) percentile penceresi."),
        ("KRIZ ESIGI",
         f"Su an sabit: {CRISIS_THRESHOLD}. Ani sok gec yakalanabiliyor. "
         "Oneri: VIX > 40 VEYA CDS > 700 gibi hard-coded ani tetikleyici ekle."),
        ("AYARLAMA BUYUKLUGU",
         "3 kademeli sabit esik (0.15 / 0.30 / 0.45). "
         "Oneri: surekli fonksiyon: adj_pt = 20 x sigmoid(dev x 8). "
         "Veya esikleri de percentile bazli yap."),
        ("VARLIK SINIRLARI",
         "Negatife dusen varlik 0'a klipleniyor, ust sinir yok. "
         "Sorun: normalize sonrasi tek varlik %60+ alabilir. "
         "Oneri: her varlik icin [min, max] kisiti. "
         "Orn: mevduat:[5,50], kripto:[0,15], hisse:[0,55]."),
        ("KORELASYON ETKISI",
         "Varliklar bagimsiz skorlaniyor. "
         "Altin ve doviz pozitif korelasyonlu: ikisi birden artinca fazla dolar riski. "
         "Oneri: skor matrisine korelasyon penaltisi ekle."),
    ]

    for baslik, metin in gelistirme:
        add(
            Paragraph(baslik, S["label"]),
            Paragraph(metin,  S["RptBody"]),
            sp(4),
        )

    # Footer notu
    add(
        sp(8), hr(C_BORDER),
        Paragraph(
            f"Bu rapor Portfolio Decision Engine v5 tarafindan otomatik uretilmistir.  "
            f"Degerlendirme tarihi: {eval_date}.  "
            "Yatirim tavsiyesi degildir.",
            S["RptCaption"]),
    )

    # ── PDF oluştur ──
    doc.build(content)
    print(f"PDF olusturuldu: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(BASE_PORTFOLIOS), default="orta_riskli")
    parser.add_argument("--output",  type=str, default=None)
    args   = parser.parse_args()

    out = args.output or f"stat_report_{args.profile}.pdf"
    try:
        build_pdf(args.profile, out)
    except FileNotFoundError as e:
        print(f"[HATA] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()