from __future__ import annotations
"""
Portfolio Decision Engine — API sunucusu

decision_engine.py'daki motoru HTTP üzerinden frontend'e açar.
Frontend artık kendi JS kopyasını hesaplamıyor; gerçek motoru çağırıyor.

Kullanım:
  python backend.py
  → http://127.0.0.1:8000  (dashboard)
  → http://127.0.0.1:8000/docs  (API dokümantasyonu)
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    """.env dosyasını (varsa) ortam değişkenlerine yükler — portfolio_data_pipeline.py
    EVDS_API_KEY'i import anında okuduğu için bu, o modülü import etmeden ÖNCE
    çalışmalı. Ek bağımlılık (python-dotenv) gerektirmeyen minimal bir parser."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(BASE_DIR / ".env")

import decision_engine as de
import portfolio_data_pipeline as pipeline

FRONTEND_FILE = BASE_DIR / "portfolio_dashboard.html"
_refresh_lock = threading.Lock()

app = FastAPI(title="Portfolio Decision Engine API", version="6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────
# Şemalar
# ─────────────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    profile: str
    date: Optional[str] = None
    amount: float = 100_000


class BacktestRequest(BaseModel):
    profile: str
    dates: List[str]
    amount: float = 100_000


# ─────────────────────────────────────────────────────────────────────────
# Statik dashboard
# ─────────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    if not FRONTEND_FILE.exists():
        raise HTTPException(404, "portfolio_dashboard.html bulunamadı")
    return FileResponse(FRONTEND_FILE)


# ─────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return {
        "assets": de.ASSETS,
        "labels": de.ASSET_LABELS,
        "base_portfolios": de.BASE_PORTFOLIOS,
        "asset_bounds": de.ASSET_BOUNDS,
        "crisis_override": de.CRISIS_OVERRIDE,
        "profile_sensitivity": de.PROFILE_SENSITIVITY,
        "neutral_threshold": de.NEUTRAL_THRESHOLD,
        "crisis_threshold": de.CRISIS_THRESHOLD,
        "vix_crisis_level": de.VIX_CRISIS_LEVEL,
        "cds_crisis_level": de.CDS_CRISIS_LEVEL,
        "rolling_window": de.ROLLING_WINDOW,
        "adjustment_levels": de.ADJ,
    }


@app.get("/api/dates")
def get_dates():
    dates = de.get_available_dates()
    if not dates:
        raise HTTPException(
            404,
            "Piyasa verisi bulunamadı. Önce portfolio_data_pipeline.py çalıştırın.",
        )
    return {"dates": dates, "min_date": dates[0], "max_date": dates[-1], "count": len(dates)}


@app.post("/api/run")
def run(req: RunRequest):
    if req.profile not in de.BASE_PORTFOLIOS:
        raise HTTPException(400, f"Geçersiz profil: {req.profile}")
    try:
        return de.run_engine(profile=req.profile, date=req.date, amount=req.amount)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    if req.profile not in de.BASE_PORTFOLIOS:
        raise HTTPException(400, f"Geçersiz profil: {req.profile}")
    if not req.dates:
        raise HTTPException(400, "En az 1 tarih gerekli")
    results = de.run_backtest(profile=req.profile, dates=req.dates, amount=req.amount)
    errored = [r for r in results if "error" in r]
    if errored and len(errored) == len(results):
        raise HTTPException(400, f"Tüm tarihler başarısız: {errored[0]['error']}")
    return {"results": results}


@app.post("/api/refresh-data")
def refresh_data():
    """
    portfolio_data_pipeline.py'yi çalıştırıp TCMB/EVDS/FRED/Yahoo/worldgovernmentbonds'tan
    veriyi yeniden çeker ve merged_aligned_daily.csv'yi günceller. Senkron `def` (async
    değil) — FastAPI bunu otomatik bir thread pool'da çalıştırır, sunucu bu sırada
    diğer istekleri (örn. /api/run) engellemez. 16 kaynağa gerçek ağ isteği attığı için
    1-3 dakika sürebilir; EVDS_API_KEY ortam değişkeni/.env'de yoksa EVDS kaynaklı
    5 seri (usdtry, eurtry, deposit_rate, cpi_index, gross_fx_reserves) başarısız olur,
    geri kalanı (FRED/Yahoo/worldgovernmentbonds) etkilenmez.
    """
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(409, "Veri çekimi zaten devam ediyor, lütfen bitmesini bekleyin.")
    try:
        results = []
        for task in pipeline.build_dataset_plan():
            try:
                res = task()
            except Exception as e:
                res = pipeline.DatasetResult(name="unknown", ok=False, source="pipeline",
                                             method="internal", error=str(e))
            results.append(res)
            if res.ok and res.rawframe is not None and not res.rawframe.empty:
                pipeline.save_raw(res.name, res.rawframe)
            if res.ok and res.dataframe is not None and not res.dataframe.empty:
                pipeline.save_processed(res.name, res.dataframe)

        merged = pipeline.merge_processed_series(results)
        if not merged.empty:
            merged.to_csv(os.path.join(pipeline.PROCESSED_DIR, "merged_aligned_daily.csv"), index=False)
        pipeline.write_logs(results)

        ok_n = sum(1 for r in results if r.ok)
        fail_n = len(results) - ok_n
        return {
            "ok": fail_n < len(results),
            "refreshed_at": datetime.now().isoformat(),
            "datasets_attempted": len(results),
            "datasets_succeeded": ok_n,
            "datasets_failed": fail_n,
            "failed": [{"dataset": r.name, "error": r.error} for r in results if not r.ok],
            "date_range": {"start": pipeline.START_DATE, "end": pipeline.END_DATE},
            "evds_key_present": bool(pipeline.EVDS_API_KEY),
        }
    finally:
        _refresh_lock.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)
