from __future__ import annotations

"""
Dynamic Portfolio Decision Engine - Türkiye Macro & Market Data Pipeline
v5 — Temizlenmiş ve düzeltilmiş sürüm

Değişiklikler (v4 → v5):
  - START_DATE 2023-01-01 olarak güncellendi (5 yıllık → ~2 yıllık, karar motoru için yeterli)
  - dxy_exact_site ve vix_exact_site pipeline'dan çıkarıldı (FRED'den zaten geliyor, duplicate)
  - turkey_10y_bond: Yahoo sembolü çalışmıyor → EVDS'e taşındı (TP.DT.TRY.10)
  - turkey_10y_inv (Investing, sadece 20 satır) → fallback olarak kalıyor ama asıl kaynak EVDS
  - turkey_cds_5y (Investing, sadece 20 satır) → pipeline'da kalıyor, kısa vadeli sinyal için yeterli
  - BASE_PORTFOLIOS tanımlandı: az/orta/cok riskli için baseline dağılımlar
"""

import io
import json
import os
import re
import time
import urllib.error
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except Exception:
    HAS_SELENIUM = False


# =============================================================================
# CONFIG
# =============================================================================
START_DATE   = os.environ.get("START_DATE",   "2020-01-01")   # v6.2: backtest için çoklu rejim (Covid çöküşü, 2021 kur krizi, 2023 sıkılaştırma)
END_DATE     = os.environ.get("END_DATE",     datetime.today().strftime("%Y-%m-%d"))
OUTPUT_ROOT  = os.environ.get("OUTPUT_ROOT",  "./portfolio_data")
EVDS_API_KEY = os.environ.get("EVDS_API_KEY", "")

RAW_DIR       = os.path.join(OUTPUT_ROOT, "raw")
PROCESSED_DIR = os.path.join(OUTPUT_ROOT, "processed")
LOG_DIR       = os.path.join(OUTPUT_ROOT, "logs")
for p in [OUTPUT_ROOT, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    os.makedirs(p, exist_ok=True)

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json"}

# =============================================================================
# BASE PORTFOLIO ALLOCATIONS (karar motorunun başlangıç noktası)
# Motor bu yüzdeleri ±5 / ±10 / ±20 pt ile dinamik olarak değiştirir.
# Kriz modunda: hisse+kripto → 0, mevduat+döviz+altın → max
# =============================================================================
BASE_PORTFOLIOS = {
    "az_riskli": {
        "mevduat":         35,   # mevduat / faiz / para piyasası fonu
        "doviz":           20,   # USD / EUR
        "altin":           25,   # gram altın / ons
        "tahvil":          12,   # tahvil / eurobond
        "yatirim_fonu":     8,   # karma / tahvil ağırlıklı fon
        "hisse":            0,
        "temettu_hisse":    0,
        "kripto":           0,
    },
    "orta_riskli": {
        "mevduat":         15,
        "doviz":           10,
        "altin":           20,
        "tahvil":           5,
        "yatirim_fonu":    10,
        "hisse":           25,
        "temettu_hisse":   15,
        "kripto":           0,
    },
    "cok_riskli": {
        "mevduat":          0,
        "doviz":            5,   # minimum koruma tamponu
        "altin":           10,
        "tahvil":           0,
        "yatirim_fonu":    10,
        "hisse":           40,
        "temettu_hisse":   20,
        "kripto":          15,
    },
}

# Kriz modu override — risk-off sinyali tetiklendiğinde uygulanır
CRISIS_OVERRIDE = {
    "mevduat":        40,
    "doviz":          30,
    "altin":          25,
    "tahvil":          5,
    "yatirim_fonu":    0,
    "hisse":           0,
    "temettu_hisse":   0,
    "kripto":          0,
}

# Aksiyon büyüklükleri (pt = yüzde puanı)
ADJUSTMENT_LEVELS = {
    "hafif":  5,   # ±5 pt
    "orta":  10,   # ±10 pt
    "guclu": 20,   # ±20 pt
    "kriz":  "override",
}

# =============================================================================
# EVDS SERİLERİ
# =============================================================================
EVDS_SERIES = {
    "usdtry":            ("TP.DK.USD.A.YTL",  1),
    "eurtry":            ("TP.DK.EUR.A.YTL",  1),
    "deposit_rate":      ("TP.KTF10",          5),
    # v6.2: TP.FG.J0, 2026-01'den sonra TCMB tarafından güncellenmeyi bırakmış
    # (muhtemelen rebaseleme/kod değişikliği) — pipeline sessizce Ocak değerini
    # Ağustos'a kadar ffill'liyordu, cpi_yoy'u yanlış (çok düşük) gösteriyordu.
    # TP.TUFE1YI.T1 aynı enflasyon dinamiğini takip eden, güncel tutulan seri.
    "cpi_index":         ("TP.TUFE1YI.T1",     5),
    "gross_fx_reserves": ("TP.AB.B1",          5),
    # v6.2: turkey_10y_bond buradan çıkarıldı — TP.DT.TRY.10 geçersiz/kaldırılmış bir
    # seri kodu (400 Bad Request), TCMB EVDS ikincil piyasa tahvil getirisi yayımlamıyor.
    # Artık worldgovernmentbonds.com'dan çekiliyor (bkz. WGB_SERIES).
}

FRED_SERIES = {
    "fed_funds":          "FEDFUNDS",
    "us10y":              "DGS10",
    "vix":                "VIXCLS",
    "broad_dollar_index": "DTWEXBGS",
}

YF_SYMBOLS = {
    "gold_ons_usd":     "GC=F",
    "btc_usd":          "BTC-USD",
    "brent_oil":        "BZ=F",
    "bist100_primary":  "XU100.IS",
    "bist100_fallback": "^XU100",
}

# v6.2: Investing.com artık istek seviyesinde (düz requests.get bile) 403 ile bilinçli
# olarak bot engelliyor — bunu aşmaya çalışmıyoruz. turkey_cds_5y ve turkey_10y_bond
# worldgovernmentbonds.com'un kendi sayfasının kullandığı genel-amaçlı JSON API'sine
# taşındı (bkz. WGB_SERIES / fetch_worldgovernmentbonds). Kimlik doğrulama gerekmiyor,
# bot koruması yok — sitenin kendi ön ucunun çağırdığı aynı public endpoint.
INVESTING_SOURCES: dict = {}

WGB_ENDPOINT = "https://www.worldgovernmentbonds.com/wp-json/common/v1/historical"
WGB_COUNTRY_TURKEY = {"SYMBOL": "13", "PAESE": "Turkey", "PAESE_UPPERCASE": "TURKEY",
                       "BANDIERA": "tr", "URL_PAGE": "turkey"}
WGB_SERIES = {
    # name: (FUNCTION, DURATA_STRING, DURATA, unit, decimal)
    "turkey_cds_5y":   ("CDS",  "5 Years",  60,  "",  2),
    "turkey_10y_bond": ("Bond", "10 Years", 120, "%", 3),
}

# TCMB politika faizi gömülü CSV (canlı kaynak başarısız olursa fallback)
TCMB_POLICY_RATE_CSV = """\
date,value
2021-01-01,17.0
2021-03-19,20.5
2021-09-24,19.5
2021-10-22,17.5
2021-11-19,16.5
2021-12-17,15.5
2022-08-19,14.5
2022-09-23,13.5
2022-10-21,12.0
2022-11-25,10.5
2023-02-24,10.0
2023-06-23,16.5
2023-07-21,19.0
2023-08-25,26.5
2023-09-22,31.5
2023-10-27,36.5
2023-11-24,41.5
2023-12-22,44.0
2024-01-26,46.5
2024-03-22,53.0
2024-12-27,49.0
2025-01-24,46.5
2025-03-07,44.0
2025-03-21,46.0
2025-04-18,49.0
2025-07-25,46.0
2025-09-12,43.5
2025-10-24,42.5
2025-12-12,41.0
2026-01-23,40.0
"""

TCMB_POLICY_RATE_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/"
    "Temel+Faaliyetler/Para+Politikasi/Merkez+Bankasi+Faiz+Oranlari/faiz-oranlari"
)


# =============================================================================
# DATA MODELS
# =============================================================================
@dataclass
class DatasetResult:
    name:      str
    ok:        bool
    source:    str
    method:    str
    dataframe: Optional[pd.DataFrame] = None
    rawframe:  Optional[pd.DataFrame] = None
    error:     Optional[str] = None
    notes:     str = ""


# =============================================================================
# CORE HELPERS
# =============================================================================
def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


def save_raw(name: str, df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        df.to_csv(os.path.join(RAW_DIR, f"{sanitize_filename(name)}.csv"), index=False)


def save_processed(name: str, df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        df.to_csv(os.path.join(PROCESSED_DIR, f"{sanitize_filename(name)}.csv"), index=False)


def _parse_flexible_date(val: str) -> pd.Timestamp:
    val = str(val).strip()
    if re.match(r"^\d{4}-\d{1,2}$", val):
        y, m = val.split("-")
        return pd.Timestamp(year=int(y), month=int(m), day=1)
    for fmt in ("%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y",
                "%d %b %Y", "%b %d, %Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(val, errors="coerce")


def _normalize_locale_number(s: str) -> str:
    """
    '.' ve ',' hangisinin ondalik ayirici oldugunu tahmin eder.
    Turkce format: 1.234,56  (nokta=binlik, virgul=ondalik)
    Ingilizce format: 1,234.56  (virgul=binlik, nokta=ondalik)
    Eskiden hep Turkce format varsayiliyordu -> Investing.com Ingilizce
    formatta '306.86' donduginde nokta binlik sanilip siliniyor,
    '30686' gibi 100x buyuk bir deger uretiyordu (turkey_cds_5y icin gorulen bug).
    """
    s = s.strip()
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            return s.replace(".", "").replace(",", ".")   # Turkce
        return s.replace(",", "")                          # Ingilizce
    if has_comma and not has_dot:
        return s.replace(",", ".")                          # Turkce ondalik
    return s   # nokta ondalik ya da tam sayi -> oldugu gibi birak


def ensure_date_value(df: pd.DataFrame, *, date_col: str, value_col: str) -> pd.DataFrame:
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"]  = out["date"].astype(str).str.strip().apply(_parse_flexible_date)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = (out.dropna(subset=["date", "value"])
              .sort_values("date")
              .drop_duplicates(subset=["date"], keep="last"))
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["date", "value"]].reset_index(drop=True)


def daily_align(df: pd.DataFrame, *, start_date: str, end_date: str,
                method: str = "ffill") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    out = out.reindex(idx)
    if method == "ffill":
        out["value"] = out["value"].ffill()
    elif method == "interpolate":
        out["value"] = out["value"].interpolate(method="time").ffill().bfill()
    out = (out.dropna(subset=["value"])
              .reset_index()
              .rename(columns={"index": "date"}))
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["date", "value"]]


def validate_nonempty(df: Optional[pd.DataFrame], name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name}: işlendikten sonra 0 satır kaldı")
    return df


# Sadece GEÇİCİ (ağ seviyesi) hatalarda tekrar dener — 400/404 gibi kalıcı
# HTTP durum hatalarında (HTTPError) retry yapmaz, direkt yükseltir; aksi halde
# yanlış bir series/sembol kodu her seferinde 3x boşa denenir.
TRANSIENT_ERRORS = (
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    TimeoutError,
    ConnectionError,
    urllib.error.URLError,   # pd.read_csv gibi urllib tabanlı çağrılar için
)
PERMANENT_ERRORS = (requests.exceptions.HTTPError, urllib.error.HTTPError)


def with_retries(fn: Callable[[], object], *, retries: int = 3, backoff: float = 2.0,
                  retry_on: tuple = TRANSIENT_ERRORS, label: str = ""):
    """
    Genel amaçlı retry sarmalayıcı — herhangi bir ağ çağrısını üstel bekleme ile
    tekrar dener. cpi_index'in tek bir chunk'ının timeout'a uğrayıp o aralığın
    sessizce eksik kalması gibi durumları önlemek için eklendi (bkz. fetch_evds_chunked).

    Not: urllib.error.HTTPError, URLError'ın alt sınıfı olduğu için retry_on'a
    dahil olsa bile PERMANENT_ERRORS içindeyse hemen yükseltilir (kalıcı 4xx/5xx
    hatalarını boşuna 3 kez denemeyi önler).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except PERMANENT_ERRORS:
            raise
        except retry_on as e:
            last_err = e
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                print(f"  [retry] {label} deneme {attempt}/{retries} başarısız ({e}), {wait:.0f}sn sonra tekrar...")
                time.sleep(wait)
    raise last_err


# =============================================================================
# A) TCMB POLİTİKA FAİZİ
# =============================================================================
def fetch_tcmb_policy_rate_live() -> pd.DataFrame:
    r = with_retries(
        lambda: requests.get(TCMB_POLICY_RATE_URL, headers={"User-Agent": BROWSER_UA}, timeout=30),
        label="TCMB policy_rate")
    r.raise_for_status()
    soup  = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Tablo bulunamadı")
    rows = table.find_all("tr")
    data = []
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) == 3 and cells[0] != "Tarih":
            data.append(cells)
    if not data:
        raise ValueError("Satır yok")
    df = pd.DataFrame(data, columns=["Tarih", "Borc_Alma", "Borc_Verme"])
    df["date"]  = pd.to_datetime(df["Tarih"], format="%d.%m.%y", errors="coerce")
    df["value"] = pd.to_numeric(df["Borc_Verme"], errors="coerce")
    return df[["date", "value"]].dropna()


def tcmb_policy_rate_dataset() -> DatasetResult:
    name = "policy_rate"
    try:
        raw    = fetch_tcmb_policy_rate_live()
        source = "TCMB (live)"
        notes  = "live"
    except Exception as e:
        print(f"  [policy_rate] Canlı çekim başarısız ({e}) → gömülü CSV")
        raw    = pd.read_csv(io.StringIO(TCMB_POLICY_RATE_CSV))
        raw["date"] = pd.to_datetime(raw["date"])
        source = "TCMB (embedded)"
        notes  = "embedded"
    try:
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        std = ensure_date_value(raw, date_col="date", value_col="value")
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method="ffill")
        std = validate_nonempty(std, name)
        return DatasetResult(name=name, ok=True, source=source, method="html_parse",
                             dataframe=std, rawframe=raw, notes=notes)
    except Exception as e:
        return DatasetResult(name=name, ok=False, source=source,
                             method="html_parse", error=str(e))


# =============================================================================
# B) EVDS
# =============================================================================
def fetch_evds(series_code: str, start_date: str, end_date: str,
               frequency: int = 1) -> pd.DataFrame:
    if not EVDS_API_KEY:
        raise ValueError("EVDS_API_KEY eksik")
    url = (
        f"https://evds3.tcmb.gov.tr/igmevdsms-dis/"
        f"series={series_code}"
        f"&startDate={pd.to_datetime(start_date).strftime('%d-%m-%Y')}"
        f"&endDate={pd.to_datetime(end_date).strftime('%d-%m-%Y')}"
        f"&type=json&aggregationTypes=avg&formulas=0&frequency={frequency}"
    )
    r = requests.get(url, headers={"key": EVDS_API_KEY, "Accept": "application/json"},
                     timeout=60)
    if "<html" in r.text.lower():
        raise ValueError(f"EVDS HTML döndü. Status={r.status_code}")
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"EVDS boş: {r.text[:150]}")
    return pd.DataFrame(items)


def fetch_evds_chunked(series_code: str, start_date: str, end_date: str,
                       chunk_days: int = 800, frequency: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """
    Döner: (birleşik veri, kalıcı olarak başarısız kalan aralıkların listesi).
    Her chunk 3 deneme hakkına sahip (with_retries) — geçici timeout'lar burada
    özümsenir. 3 denemeden sonra hâlâ başarısızsa aralık failed_ranges'e eklenir
    ve DatasetResult.notes üzerinden dışarı sızdırılır (eskiden sessizce yutulup
    "ok=True" ile eksik veri üretiyordu — bkz. cpi_index 2023-01→2025-03 kaybı).
    """
    start, end, frames, current = (pd.to_datetime(start_date), pd.to_datetime(end_date),
                                   [], pd.to_datetime(start_date))
    failed_ranges: List[str] = []
    while current < end:
        ce = min(current + pd.Timedelta(days=chunk_days), end)
        label = f"{series_code} {current.date()}→{ce.date()}"
        try:
            frames.append(with_retries(
                lambda c=current, e=ce: fetch_evds(series_code, c.strftime("%Y-%m-%d"),
                                                    e.strftime("%Y-%m-%d"), frequency=frequency),
                label=label))
        except Exception as e:
            print(f"  [EVDS] {label} kalıcı başarısız (tüm denemeler tükendi): {e}")
            failed_ranges.append(f"{current.date()}..{ce.date()}")
        current = ce + pd.Timedelta(days=1)
    if not frames:
        raise ValueError("Chunk veri yok")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.apply(
        lambda col: col.map(lambda x: str(x) if isinstance(x, (dict, list)) else x))
    return combined.drop_duplicates(), failed_ranges


def evds_single_series_dataset(name: str, series_code: str,
                                frequency: int = 1,
                                fill_method: str = "ffill") -> DatasetResult:
    try:
        raw, failed_ranges = fetch_evds_chunked(series_code, START_DATE, END_DATE, frequency=frequency)
        dcol  = next((c for c in raw.columns if c.upper() == "TARIH"), raw.columns[0])
        vcols = [c for c in raw.columns if c not in {dcol, "UNIXTIME"}]
        if not vcols:
            raise ValueError("Değer sütunu yok")
        std = ensure_date_value(raw, date_col=dcol, value_col=vcols[0])
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method=fill_method)
        std = validate_nonempty(std, name)
        notes = f"EKSIK ARALIK (kalıcı hata): {'; '.join(failed_ranges)}" if failed_ranges else ""
        return DatasetResult(name=name, ok=True, source="EVDS", method="api",
                             dataframe=std, rawframe=raw, notes=notes)
    except Exception as e:
        return DatasetResult(name=name, ok=False, source="EVDS", method="api", error=str(e))


# =============================================================================
# C) FRED
# =============================================================================
def fred_dataset(name: str, series_id: str, fill_method: str = "ffill") -> DatasetResult:
    try:
        df = with_retries(
            lambda: pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"),
            label=f"FRED {series_id}")
        if df.empty:
            raise ValueError("Boş")
        df.columns = ["date", "value"]
        df["value"] = df["value"].replace(".", float("nan"))
        std = ensure_date_value(df, date_col="date", value_col="value")
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method=fill_method)
        std = validate_nonempty(std, name)
        return DatasetResult(name=name, ok=True, source="FRED", method="csv_endpoint",
                             dataframe=std, rawframe=df)
    except Exception as e:
        return DatasetResult(name=name, ok=False, source="FRED",
                             method="csv_endpoint", error=str(e))


# =============================================================================
# D) YAHOO FINANCE
# =============================================================================
def yfinance_dataset(name: str, symbol: str, price_col: str = "Close") -> DatasetResult:
    try:
        if yf is None:
            raise ImportError("yfinance kurulu değil")
        raw = with_retries(
            lambda: yf.download(symbol, start=START_DATE, end=END_DATE,
                                progress=False, auto_adjust=False),
            label=f"Yahoo {symbol}")
        if raw is None or raw.empty:
            raise ValueError(f"Veri yok: {symbol}")
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        if price_col not in raw.columns:
            raise ValueError(f"Sütun yok: {price_col}")
        std = ensure_date_value(raw, date_col="Date", value_col=price_col)
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method="ffill")
        std = validate_nonempty(std, name)
        return DatasetResult(name=name, ok=True, source="Yahoo Finance", method="yfinance",
                             dataframe=std, rawframe=raw, notes=f"symbol={symbol}")
    except Exception as e:
        return DatasetResult(name=name, ok=False, source="Yahoo Finance",
                             method="yfinance", error=str(e))


def yfinance_with_fallback(name: str, symbols: List[str]) -> DatasetResult:
    last_error = None
    for sym in symbols:
        res = yfinance_dataset(name=name, symbol=sym)
        if res.ok:
            return res
        last_error = res.error
    return DatasetResult(name=name, ok=False, source="Yahoo Finance", method="yfinance",
                         error=last_error or "Tüm semboller başarısız")


# =============================================================================
# E) INVESTING.COM — Selenium ile tam tarih aralığı
# =============================================================================
def _make_driver():
    if not HAS_SELENIUM:
        raise ImportError("selenium kurulu değil")
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(f"user-agent={BROWSER_UA}")
    return webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()), options=opts)


def _scrape_investing_page(url: str, sleep_sec: int = 5) -> pd.DataFrame:
    """Selenium ile sayfadaki tüm tabloyu çeker."""
    driver = _make_driver()
    try:
        driver.get(url)
        time.sleep(sleep_sec)
        try:
            inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
            if len(inputs) >= 2:
                s = pd.to_datetime(START_DATE).strftime("%d.%m.%Y")
                e = pd.to_datetime(END_DATE).strftime("%d.%m.%Y")
                for inp, val in zip(inputs[:2], [s, e]):
                    inp.clear()
                    inp.send_keys(val)
                inputs[1].send_keys(Keys.RETURN)
                time.sleep(sleep_sec)
        except Exception:
            pass
        html   = driver.page_source
        tables = pd.read_html(io.StringIO(html))
        if not tables:
            raise ValueError("Tablo bulunamadı")
        return max(tables, key=len)
    finally:
        driver.quit()


def investing_dataset(name: str, url: str,
                       date_candidates: List[str],
                       value_candidates: List[str],
                       fill_method: str = "ffill") -> DatasetResult:
    def _parse(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]
        dc = next((c for c in df.columns
                   if any(v.lower() in c.lower() for v in date_candidates)), None)
        vc = next((c for c in df.columns
                   if any(v.lower() in c.lower() for v in value_candidates)), None)
        if dc is None or vc is None:
            raise ValueError(f"Sütun yok. Mevcut: {list(df.columns)}")
        tmp = df.copy()
        tmp[vc] = tmp[vc].astype(str).str.replace("%", "", regex=False).str.strip()
        tmp[vc] = tmp[vc].apply(_normalize_locale_number)
        return ensure_date_value(tmp, date_col=dc, value_col=vc)

    raw = None
    try:
        tables = pd.read_html(url, flavor="lxml")
        if tables:
            raw = max(tables, key=len)
    except Exception:
        pass

    if raw is None or len(raw) < 30:
        try:
            raw = _scrape_investing_page(url)
        except Exception as se:
            if raw is None:
                return DatasetResult(name=name, ok=False, source=url,
                                     method="investing", error=str(se))

    try:
        std = _parse(raw)
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method=fill_method)
        std = validate_nonempty(std, name)
        return DatasetResult(name=name, ok=True, source=url, method="investing",
                             dataframe=std, rawframe=raw,
                             notes=f"raw_rows={len(raw)}")
    except Exception as e:
        return DatasetResult(name=name, ok=False, source=url,
                             method="investing", error=str(e))


# =============================================================================
# F) WORLDGOVERNMENTBONDS.COM — CDS + tahvil getirisi (Investing.com/EVDS yerine)
# =============================================================================
def fetch_worldgovernmentbonds(function: str, durata_string: str, durata: int,
                                country: dict = WGB_COUNTRY_TURKEY,
                                unit: str = "", decimal: int = 2) -> pd.DataFrame:
    """
    worldgovernmentbonds.com kendi sayfasını (ör. /cds-historical-data/turkey/5-years/)
    JS ile render ederken bu genel-amaçlı JSON API'yi çağırıyor. Kimlik doğrulama
    gerekmiyor, bot koruması yok — sadece kendi sitesinden gelen isteklerle sınırlamak
    için Origin/Referer kontrol ediyor, o yüzden onları set ediyoruz.
    """
    payload = {
        "GLOBALVAR": {
            "JS_VARIABLE": "jsGlobalVars", "FUNCTION": function, "DOMESTIC": True,
            "ENDPOINT": WGB_ENDPOINT, "DATE_RIF": "2099-12-31",
            "OBJ": {"UNIT": unit, "DECIMAL": decimal, "UNIT_DELTA": "%", "DECIMAL_DELTA": 2},
            "COUNTRY1": country, "COUNTRY2": None,
            "OBJ1": {"DURATA_STRING": durata_string, "DURATA": durata},
            "OBJ2": None,
        }
    }
    headers = {
        "User-Agent": BROWSER_UA, "Content-Type": "application/json",
        "Origin": "https://www.worldgovernmentbonds.com",
        "Referer": "https://www.worldgovernmentbonds.com/",
    }

    def _do():
        r = requests.post(WGB_ENDPOINT, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r

    r = with_retries(_do, label=f"WGB {function} {durata_string}")
    data = r.json()
    if not data.get("success"):
        raise ValueError(f"WGB başarısız yanıt: {str(data)[:150]}")
    quote = data.get("result", {}).get("quote", {})
    if not quote:
        raise ValueError("WGB boş sonuç döndü")
    return pd.DataFrame([{"date": v["DATA_VAL"], "value": v["CLOSE_VAL"]} for v in quote.values()])


def wgb_dataset(name: str, function: str, durata_string: str, durata: int,
                 unit: str = "", decimal: int = 2, fill_method: str = "ffill") -> DatasetResult:
    try:
        raw = fetch_worldgovernmentbonds(function, durata_string, durata, unit=unit, decimal=decimal)
        std = ensure_date_value(raw, date_col="date", value_col="value")
        std = daily_align(std, start_date=START_DATE, end_date=END_DATE, method=fill_method)
        std = validate_nonempty(std, name)
        return DatasetResult(name=name, ok=True, source="worldgovernmentbonds.com",
                             method="json_api", dataframe=std, rawframe=raw)
    except Exception as e:
        return DatasetResult(name=name, ok=False, source="worldgovernmentbonds.com",
                             method="json_api", error=str(e))


# =============================================================================
# PIPELINE SPEC
# =============================================================================
def build_dataset_plan() -> List[Callable[[], DatasetResult]]:
    plan: List[Callable[[], DatasetResult]] = []

    # 1. TCMB politika faizi
    plan.append(tcmb_policy_rate_dataset)

    # 2. EVDS
    for ds_name, (code, freq) in EVDS_SERIES.items():
        n, c, f = ds_name, code, freq
        plan.append(lambda n=n, c=c, f=f:
                    evds_single_series_dataset(n, c, frequency=f, fill_method="ffill"))

    # 3. FRED
    for ds_name, fcode in FRED_SERIES.items():
        n, c = ds_name, fcode
        plan.append(lambda n=n, c=c: fred_dataset(n, c, fill_method="ffill"))

    # 4. Yahoo Finance
    plan.append(lambda: yfinance_dataset("gold_ons_usd", YF_SYMBOLS["gold_ons_usd"]))
    plan.append(lambda: yfinance_dataset("btc_usd",      YF_SYMBOLS["btc_usd"]))
    plan.append(lambda: yfinance_dataset("brent_oil",    YF_SYMBOLS["brent_oil"]))
    plan.append(lambda: yfinance_with_fallback(
        "bist100", [YF_SYMBOLS["bist100_primary"], YF_SYMBOLS["bist100_fallback"]]))

    # 5. Investing.com — artık kullanılmıyor (site istek seviyesinde 403 ile bilinçli
    # bot engelliyor). INVESTING_SOURCES boş bırakıldı, aşağıdaki döngü no-op.
    dc = ["Date", "Tarih", "Zaman"]
    vc = ["Price", "Fiyat", "Son", "Şimdi"]
    for ds_name, url in INVESTING_SOURCES.items():
        n, u = ds_name, url
        plan.append(lambda n=n, u=u:
                    investing_dataset(n, u, date_candidates=dc, value_candidates=vc))

    # 6. worldgovernmentbonds.com — turkey_cds_5y + turkey_10y_bond
    for ds_name, (fn, durata_str, durata, unit, decimal) in WGB_SERIES.items():
        n, f_, ds_, d_, u_, dec_ = ds_name, fn, durata_str, durata, unit, decimal
        plan.append(lambda n=n, f_=f_, ds_=ds_, d_=d_, u_=u_, dec_=dec_:
                    wgb_dataset(n, f_, ds_, d_, unit=u_, decimal=dec_))

    return plan


# =============================================================================
# MERGE & LOGS
# =============================================================================
def merge_processed_series(results: List[DatasetResult]) -> pd.DataFrame:
    merged = None
    for r in results:
        if not r.ok or r.dataframe is None or r.dataframe.empty:
            continue
        tmp    = r.dataframe.copy().rename(columns={"value": r.name})
        merged = tmp if merged is None else merged.merge(tmp, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["date"])
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").set_index("date")
    merged = merged.ffill().dropna(how="all").reset_index()
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def write_logs(results: List[DatasetResult]) -> None:
    summary, failed = [], []
    for r in results:
        summary.append({"dataset": r.name, "ok": r.ok, "source": r.source,
                         "method": r.method,
                         "rows": 0 if r.dataframe is None else len(r.dataframe),
                         "error": r.error, "notes": r.notes})
        if not r.ok:
            failed.append({"dataset": r.name, "source": r.source,
                            "method": r.method, "reason": r.error})
    pd.DataFrame(summary).to_csv(os.path.join(LOG_DIR, "fetch_summary.csv"), index=False)
    pd.DataFrame(failed).to_csv(os.path.join(LOG_DIR, "failed_sources.csv"), index=False)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print(f"Pipeline başlatıldı | {START_DATE} → {END_DATE}")
    print(f"Toplam dataset: {len(build_dataset_plan())}")
    print("-" * 60)

    results: List[DatasetResult] = []
    for task in build_dataset_plan():
        try:
            res = task()
        except Exception as e:
            res = DatasetResult(name="unknown", ok=False, source="pipeline",
                                method="internal", error=str(e))
        results.append(res)
        status = "✓" if res.ok else "✗"
        rows   = len(res.dataframe) if res.ok and res.dataframe is not None else 0
        print(f"  [{status}] {res.name:<25} {rows:>5} satır | {res.source}")
        if res.ok and res.rawframe is not None and not res.rawframe.empty:
            save_raw(res.name, res.rawframe)
        if res.ok and res.dataframe is not None and not res.dataframe.empty:
            save_processed(res.name, res.dataframe)

    merged = merge_processed_series(results)
    if not merged.empty:
        merged.to_csv(os.path.join(PROCESSED_DIR, "merged_aligned_daily.csv"), index=False)

    write_logs(results)

    # Base portfolio config'i de kaydet
    with open(os.path.join(OUTPUT_ROOT, "base_portfolios.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_portfolios": BASE_PORTFOLIOS,
            "crisis_override": CRISIS_OVERRIDE,
            "adjustment_levels_pt": ADJUSTMENT_LEVELS,
        }, f, ensure_ascii=False, indent=2)

    ok_n   = sum(1 for r in results if r.ok)
    fail_n = sum(1 for r in results if not r.ok)
    meta = {
        "generated_at":       datetime.now().isoformat(),
        "start_date":         START_DATE,
        "end_date":           END_DATE,
        "output_root":        os.path.abspath(OUTPUT_ROOT),
        "evds_key_present":   bool(EVDS_API_KEY),
        "datasets_attempted": len(results),
        "datasets_succeeded": ok_n,
        "datasets_failed":    fail_n,
    }
    with open(os.path.join(LOG_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"Başarılı : {ok_n} / {len(results)}")
    print(f"Başarısız: {fail_n} / {len(results)}")
    if not merged.empty:
        cols = [c for c in merged.columns if c != "date"]
        print(f"Sütunlar : {', '.join(cols)}")
        print(f"Satırlar : {len(merged):,}")
    if fail_n:
        print(f"Hatalar  : {os.path.join(os.path.abspath(LOG_DIR), 'failed_sources.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()