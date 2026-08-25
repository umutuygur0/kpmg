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

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import decision_engine as de

BASE_DIR = Path(__file__).parent
FRONTEND_FILE = BASE_DIR / "portfolio_dashboard.html"

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)
