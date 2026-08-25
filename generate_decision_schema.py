"""
Decision Engine Karar Şeması — Sunum Görseli
Çalıştır: python generate_decision_schema.py
Çıktı: decision_schema.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ─── Renkler ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0F1117",
    "panel":    "#1A1D2E",
    "border":   "#2E3250",
    "raw":      "#2563EB",   # mavi – ham veri
    "derived":  "#7C3AED",   # mor – türetilmiş sinyal
    "score":    "#059669",   # yeşil – skor
    "adj":      "#D97706",   # turuncu – ayarlama
    "crisis":   "#DC2626",   # kırmızı – kriz
    "output":   "#0891B2",   # camgöbeği – çıktı
    "text":     "#F1F5F9",
    "subtext":  "#94A3B8",
    "arrow":    "#475569",
    "weight":   "#FCD34D",
    "positive": "#34D399",
    "negative": "#F87171",
}

fig = plt.figure(figsize=(26, 18), facecolor=C["bg"])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 26)
ax.set_ylim(0, 18)
ax.set_facecolor(C["bg"])
ax.axis("off")

# ─── Yardımcı fonksiyonlar ──────────────────────────────────────────────────
def box(ax, x, y, w, h, color, alpha=0.85, radius=0.15):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       linewidth=1.2, edgecolor=color,
                       facecolor=color + "28", zorder=3)
    ax.add_patch(p)

def label(ax, x, y, text, size=8.5, color=C["text"], bold=False, ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, color=color,
            fontweight="bold" if bold else "normal",
            ha=ha, va=va, zorder=5, wrap=False)

def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=1.2, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=4)

def section_title(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=10, color=color, fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color+"22",
                      edgecolor=color, linewidth=1.2))

# ═══════════════════════════════════════════════════════════════════════════════
# BAŞLIK
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(13, 17.5, "PORTFÖY KARAR MOTORU — Karar Akış Şeması",
        fontsize=16, color=C["text"], fontweight="bold", ha="center", va="center")
ax.text(13, 17.1, "Her ok: hangi sinyal, hangi varlık kararını nasıl etkiler",
        fontsize=9, color=C["subtext"], ha="center", va="center")

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 1 — HAM VERİ GİRDİLERİ  (x≈1.0)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 2.0, 16.6, "① HAM VERİ", C["raw"])

raw_inputs = [
    ("policy_rate",    "Politika Faizi (%)"),
    ("cpi_index",      "CPI Endeksi"),
    ("usdtry",         "USD/TRY"),
    ("bist100",        "BIST 100"),
    ("gold_ons_usd",   "Altın (USD/oz)"),
    ("vix",            "VIX"),
    ("turkey_cds_5y",  "Türkiye CDS 5Y"),
    ("us10y",          "ABD 10Y Faiz"),
]

raw_y_start = 15.8
raw_step    = 0.85
raw_x       = 0.4
raw_w       = 3.0
raw_h       = 0.62

for i, (key, name) in enumerate(raw_inputs):
    yy = raw_y_start - i * raw_step
    box(ax, raw_x, yy - raw_h/2, raw_w, raw_h, C["raw"])
    label(ax, raw_x + raw_w/2, yy, f"{name}", size=8, color=C["text"])
    label(ax, raw_x + raw_w/2, yy - 0.2, f"[{key}]", size=6.5, color=C["subtext"])

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 2 — TÜRETİLMİŞ SİNYALLER  (x≈5.0)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 6.5, 16.6, "② TÜRETİLMİŞ SİNYALLER", C["derived"])

derived = [
    ("real_rate",        "Reel Faiz",           "policy_rate − cpi_yoy",       [0,1]),
    ("fx_stress",        "FX Stres",            "std(USDTRY,30) / mean,  0-1", [2]),
    ("bist_momentum",    "BIST Momentum",       "(BIST−MA20)/MA20×100, kırp±30",[3]),
    ("gold_real_return", "Altın Reel Getiri",   "gold_TRY yıllık % − CPI %",   [0,1,2,4]),
    ("vix_level",        "VIX Seviyesi",        "doğrudan",                     [5]),
    ("cds_level",        "CDS Seviyesi",        "doğrudan",                     [6]),
    ("risk_score",       "Bileşik Risk Skoru",  "VIX×.20+FX×.25+rr×.20\n+CDS×.20+BIST×.15", [0,1,2,3,4,5,6]),
]

der_y_start = 15.8
der_step    = 1.30
der_x       = 4.8
der_w       = 3.4
der_h       = 0.80

for i, (key, name, formula, src_idxs) in enumerate(derived):
    yy = der_y_start - i * der_step
    color = C["crisis"] if key == "risk_score" else C["derived"]
    box(ax, der_x, yy - der_h/2, der_w, der_h, color)
    label(ax, der_x + der_w/2, yy + 0.12, name, size=8.5, bold=True, color=C["text"])
    label(ax, der_x + der_w/2, yy - 0.18, formula, size=6.5, color=C["subtext"])
    # oklar raw → derived
    for si in src_idxs:
        src_yy = raw_y_start - si * raw_step
        arrow(ax, raw_x + raw_w, src_yy, der_x, yy, color=C["raw"]+"88")

# ═══════════════════════════════════════════════════════════════════════════════
# KRİZ BLOĞU (derived'dan ayrı)
# ═══════════════════════════════════════════════════════════════════════════════
# Kriz kutusu
section_title(ax, 6.5, 0.8, "⚠ KRİZ KONTROL", C["crisis"])
box(ax, 4.8, 1.2, 3.4, 2.2, C["crisis"])
label(ax, 6.5, 3.1, "KRİZ TETIK", size=9, bold=True, color=C["crisis"])
label(ax, 6.5, 2.75,"HARD: VIX > 40  VEYA  CDS > 700", size=7.5, color=C["text"])
label(ax, 6.5, 2.45,"SOFT: risk_score ≥ 0.80", size=7.5, color=C["text"])
label(ax, 6.5, 2.0, "KRİZ OVERRIDE:", size=8, bold=True, color=C["crisis"])
label(ax, 6.5, 1.72,"Mevduat 40% | Döviz 30%", size=7.5, color=C["text"])
label(ax, 6.5, 1.48,"Altın 25%   | Tahvil  5%", size=7.5, color=C["text"])

# risk_score → kriz kutu
risk_idx = 6   # derived listesinde risk_score = index 6
risk_yy  = der_y_start - risk_idx * der_step
arrow(ax, der_x + der_w/2, risk_yy - der_h/2, 6.5, 3.4, color=C["crisis"], lw=1.8)

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 3 — VARLIK SKORLARI  (x≈9.6)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 11.3, 16.6, "③ VARLIK SKORLARI  (0–1)", C["score"])

assets = [
    ("mevduat",       "Mevduat",        "reel_faiz×0.6 + risk×0.4",           ["real_rate","risk_score"]),
    ("doviz",         "Döviz",          "fx_stres×0.5 + risk×0.3 + (1−reel)×0.2", ["fx_stress","risk_score","real_rate"]),
    ("altin",         "Altın",          "risk×0.5 + altın_reel×0.3 + fx×0.2",["risk_score","gold_real_return","fx_stress"]),
    ("tahvil",        "Tahvil",         "reel×0.5 + (1−risk)×0.3 + us10y⁻¹×0.2",["real_rate","risk_score","us10y"]),
    ("yatirim_fonu",  "Yatırım Fonu",   "nötr(BIST)×0.4 + (1−risk)×0.3 + reel×0.3",["bist_momentum","risk_score","real_rate"]),
    ("hisse",         "Hisse",          "BIST_mom×0.5 + (1−risk)×0.5",       ["bist_momentum","risk_score"]),
    ("temettu_hisse", "Temettü Hisse",  "(1−reel)×0.4 + (1−risk)×0.4 + BIST×0.2",["real_rate","risk_score","bist_momentum"]),
    ("kripto",        "Kripto",         "(1−risk)×0.6 + us10y⁻¹×0.4",        ["risk_score","us10y"]),
]

score_y_start = 15.8
score_step    = 1.05
score_x       = 9.6
score_w       = 3.6
score_h       = 0.78

# derived key → y koordinatı
der_key_y = {d[0]: der_y_start - i*der_step for i, d in enumerate(derived)}
# US10Y ayrı: raw listesinde index 7
us10y_raw_y = raw_y_start - 7 * raw_step

for i, (key, name, formula, dep_keys) in enumerate(assets):
    yy = score_y_start - i * score_step
    box(ax, score_x, yy - score_h/2, score_w, score_h, C["score"])
    label(ax, score_x + score_w/2, yy + 0.14, name, size=9, bold=True, color=C["text"])
    label(ax, score_x + score_w/2, yy - 0.18, formula, size=6.2, color=C["subtext"])

    for dk in dep_keys:
        if dk == "us10y":
            src_y = us10y_raw_y
            src_x = raw_x + raw_w
        elif dk in der_key_y:
            src_y = der_key_y[dk]
            src_x = der_x + der_w
        else:
            continue
        arrow(ax, src_x, src_y, score_x, yy, color=C["derived"]+"66")

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 4 — AYARLAMA MANTIĞI  (x≈14.0)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 15.5, 16.6, "④ AYARLAMA KURALLARI", C["adj"])

adj_x = 13.8
adj_y = 15.0
adj_w = 3.4

box(ax, adj_x, adj_y, adj_w, 5.2, C["adj"])
label(ax, adj_x + adj_w/2, adj_y + 5.0, "score → ±puan (pt)", size=9, bold=True, color=C["adj"])
label(ax, adj_x + adj_w/2, adj_y + 4.6, "deviation = score − 0.5", size=8, color=C["text"])

rules = [
    ("|dev| < 0.08",          "Nötr",   "   0 pt",  C["subtext"]),
    ("0.08 ≤ |dev| < 0.22",   "Hafif",  " ±5 pt",   C["positive"]),
    ("0.22 ≤ |dev| < 0.38",   "Orta",   "±10 pt",   C["weight"]),
    ("|dev| ≥ 0.38",           "Güçlü",  "±20 pt",   C["negative"]),
]
for ri, (cond, label_str, pt, col) in enumerate(rules):
    ry = adj_y + 3.7 - ri * 0.72
    label(ax, adj_x + adj_w/2, ry + 0.18, cond, size=7.5, color=C["subtext"])
    label(ax, adj_x + adj_w/2, ry - 0.06,
          f"→ {label_str}: {pt}", size=8.5, bold=True, color=col)

label(ax, adj_x + adj_w/2, adj_y + 0.55,
      "× Profil Duyarlılığı", size=8, bold=True, color=C["weight"])
label(ax, adj_x + adj_w/2, adj_y + 0.25,
      "Az=×0.6  |  Orta=×1.0  |  Çok=×1.4", size=7.5, color=C["text"])

# Skor kutularından adj'a oklar
for i in range(len(assets)):
    yy = score_y_start - i * score_step
    arrow(ax, score_x + score_w, yy, adj_x, adj_y + 2.6, color=C["score"]+"55")

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 5 — SINIR KONTROLÜ  (x≈18.0)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 19.5, 16.6, "⑤ SINIR & NORMALİZASYON", C["output"])

bounds_x = 18.0
bounds_y = 12.5
bounds_w = 3.2

box(ax, bounds_x, bounds_y, bounds_w, 3.7, C["output"])
label(ax, bounds_x + bounds_w/2, bounds_y + 3.45, "Varlık Sınırları (ASSET_BOUNDS)", size=8.5, bold=True, color=C["output"])

bounds_data = [
    ("Mevduat",       "min %5   max %55"),
    ("Döviz",         "min %0   max %35"),
    ("Altın",         "min %5   max %40"),
    ("Tahvil",        "min %0   max %25"),
    ("Yat. Fonu",     "min %0   max %20"),
    ("Hisse",         "min %0   max %55"),
    ("Tem. Hisse",    "min %0   max %30"),
    ("Kripto",        "min %0   max %15"),
]
for bi, (asset, rng) in enumerate(bounds_data):
    by = bounds_y + 3.05 - bi * 0.38
    label(ax, bounds_x + 0.15, by, f"{asset:<12} {rng}", size=6.5,
          color=C["text"], ha="left")

# normalize kutusu
norm_x = 18.0
norm_y = 11.0
norm_w = 3.2
box(ax, norm_x, norm_y, norm_w, 1.0, C["output"])
label(ax, norm_x + norm_w/2, norm_y + 0.65, "Normalize", size=8.5, bold=True, color=C["output"])
label(ax, norm_x + norm_w/2, norm_y + 0.30, "Σ → 100%  (en büyük varlığa fark)", size=7, color=C["text"])

arrow(ax, adj_x + adj_w, adj_y + 2.6, bounds_x, bounds_y + 1.8, color=C["adj"])
arrow(ax, norm_x + norm_w/2, norm_y, norm_x + norm_w/2, norm_y - 0.4, color=C["output"])

# ═══════════════════════════════════════════════════════════════════════════════
# SÜTUN 6 — ÇIKIŞ  (x≈22.0)
# ═══════════════════════════════════════════════════════════════════════════════
section_title(ax, 23.8, 16.6, "⑥ ÇIKTI", C["output"])

out_x = 22.0
out_y_base = 14.5

profiles = [
    ("Az Riskli  (Koruyucu)",  "Mevduat 35 | Döviz 20\nAltın 25 | Tahvil 12\nFon 8",       "×0.6",  C["positive"]),
    ("Orta Riskli (Dengeli)", "Mevduat 15 | Döviz 10\nAltın 20 | Tahvil 5\nHisse 25 | Tem.15", "×1.0", C["weight"]),
    ("Çok Riskli (Agresif)",  "Hisse 40 | Tem.Hisse 20\nKripto 15 | Döviz 5\nAltın 10", "×1.4", C["negative"]),
]

for pi, (pname, alloc, sens, col) in enumerate(profiles):
    py = out_y_base - pi * 2.6
    box(ax, out_x, py - 1.4, 3.7, 1.9, col)
    label(ax, out_x + 3.7/2, py + 0.28, pname, size=8, bold=True, color=col)
    label(ax, out_x + 3.7/2, py - 0.1, f"Duyarlılık {sens}", size=7.5, color=C["weight"])
    label(ax, out_x + 3.7/2, py - 0.55, alloc, size=6.5, color=C["text"])
    arrow(ax, bounds_x + bounds_w, norm_y + 0.5, out_x, py - 0.45, color=C["output"]+"88")

# Kriz → çıktı
arrow(ax, der_x + der_w, der_key_y["risk_score"], out_x + 1.0, out_y_base - 5.6,
      color=C["crisis"], lw=1.5)
box(ax, out_x, out_y_base - 6.4, 3.7, 1.2, C["crisis"])
label(ax, out_x + 3.7/2, out_y_base - 5.95, "KRİZ OVERRIDE", size=9, bold=True, color=C["crisis"])
label(ax, out_x + 3.7/2, out_y_base - 6.25, "Mevduat 40 | Döviz 30 | Altın 25 | Tahvil 5", size=6.8, color=C["text"])

# ═══════════════════════════════════════════════════════════════════════════════
# KUTU ALT — Sinyal → Karar ÖZET Tablosu
# ═══════════════════════════════════════════════════════════════════════════════
# Alt bölüm: sinyal → karar çiftleri
tbl_x  = 0.3
tbl_y  = 3.8
tbl_w  = 25.4
tbl_h  = 2.9

box(ax, tbl_x, tbl_y, tbl_w, tbl_h, C["score"], alpha=0.5)
label(ax, tbl_x + tbl_w/2, tbl_y + tbl_h - 0.22,
      "SINYAL → KARAR EŞLEŞMELERİ  (örnekler)",
      size=10, bold=True, color=C["score"])

signal_decisions = [
    # (sinyal, yön, hangi varlık etkilenir, nasıl)
    ("Faiz ↑ (reel>5%)",       "→",  "Mevduat ↑   Tahvil ↑   Temettü ↓   Kripto ↓"),
    ("Faiz ↓ (reel<-5%)",      "→",  "Mevduat ↓   Tahvil ↓   Temettü ↑   Döviz ↑"),
    ("FX Stres ↑",             "→",  "Döviz ↑     Altın ↑    Mevduat ↑"),
    ("BIST Mom ↑ (>+3%)",      "→",  "Hisse ↑     Temettü ↑  Fon nötre yakın"),
    ("BIST Mom ↓ (<-3%)",      "→",  "Hisse ↓     Temettü ↓  Mevduat ↑"),
    ("Risk ↑ (>0.65)",         "→",  "Mevduat ↑   Döviz ↑    Altın ↑    Hisse ↓   Kripto ↓"),
    ("Risk ↓ (<0.35)",         "→",  "Hisse ↑     Temettü ↑  Kripto ↑   Mevduat ↓"),
    ("VIX>40 / CDS>700",       "→",  "KRİZ OVERRIDE: Mevduat 40% + Döviz 30% + Altın 25% + Tahvil 5%"),
]

cols_per_row = 4
row_h = 0.52
for i, (sig, arr, dec) in enumerate(signal_decisions):
    col = i % cols_per_row
    row = i // cols_per_row
    cx  = tbl_x + 0.3 + col * (tbl_w / cols_per_row)
    cy  = tbl_y + tbl_h - 0.55 - row * row_h
    color = C["crisis"] if "KRİZ" in dec else C["adj"]
    label(ax, cx, cy + 0.05, sig, size=7.5, bold=True, color=color, ha="left")
    label(ax, cx, cy - 0.22, dec, size=6.5, color=C["text"], ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C["raw"],     "Ham Veri"),
    (C["derived"], "Türetilmiş Sinyal"),
    (C["score"],   "Varlık Skoru"),
    (C["adj"],     "Ayarlama Kuralı"),
    (C["crisis"],  "Kriz Yolu"),
    (C["output"],  "Çıktı / Sınır"),
]
lx, ly = 0.4, 0.55
for li, (col, lbl) in enumerate(legend_items):
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx + li * 4.1, ly - 0.18), 0.45, 0.36,
        boxstyle="round,pad=0.05", facecolor=col+"44", edgecolor=col, linewidth=1.2, zorder=5))
    label(ax, lx + li * 4.1 + 0.7, ly, lbl, size=7.5, color=col, ha="left")

plt.tight_layout(pad=0)
out_path = "decision_schema.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
print(f"Kaydedildi: {out_path}")
plt.close()
