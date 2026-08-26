from __future__ import annotations
"""
optimizer.py (v7 — kısıt-native mean-variance motoru) için yapısal doğrulama.

decision_engine.py'nin FIX-7/8/9 testlerinin (tests/test_decision_engine.py)
v7 karşılığı — AYNI invaryantlar (toplam=%100, sınır ihlali yok) burada da
garanti altına alınıyor. optimize_portfolio() gerçek kovaryans matrisine
ihtiyaç duyduğundan (get_covariance_matrix() portfolio_data/ okuyor), bu
dosyadaki testler veri mevcut değilse atlanır (decision_engine'in saf
sentetik testlerinin aksine).

Çalıştır:  python -m pytest tests/test_optimizer.py -v
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import decision_engine as de

DATA_AVAILABLE = de.MERGED_CSV.exists()
pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="portfolio_data/processed/merged_aligned_daily.csv yok")

if DATA_AVAILABLE:
    import optimizer as opt

ALL_PROFILES = list(de.BASE_PORTFOLIOS.keys())


def random_scores(seed: int) -> dict:
    rng = random.Random(seed)
    return {a: rng.uniform(0.0, 1.0) for a in de.ASSETS}


class TestOptimizePortfolio:

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    @pytest.mark.parametrize("seed", range(15))
    def test_sums_to_exactly_100(self, profile, seed):
        scores = random_scores(seed)
        sens = de.PROFILE_SENSITIVITY[profile]
        final = opt.optimize_portfolio(scores, sens)
        assert sum(final.values()) == pytest.approx(100.0, abs=0.01)

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    @pytest.mark.parametrize("seed", range(15))
    def test_never_breaches_asset_bounds(self, profile, seed):
        scores = random_scores(seed)
        sens = de.PROFILE_SENSITIVITY[profile]
        final = opt.optimize_portfolio(scores, sens)
        for asset, value in final.items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01, f"{profile}/seed={seed}: {asset}={value} sınır dışı"

    def test_no_negative_allocations(self):
        for profile in ALL_PROFILES:
            for seed in range(10):
                scores = random_scores(seed)
                final = opt.optimize_portfolio(scores, de.PROFILE_SENSITIVITY[profile])
                assert all(v >= -0.01 for v in final.values())

    def test_turnover_penalty_reduces_movement_from_prev(self):
        """Aynı skorlarla, prev_weights'e yakın kalması beklenir (ceza etkisi)."""
        scores = random_scores(seed=1)
        sens = de.PROFILE_SENSITIVITY["orta_riskli"]
        prev = de.BASE_PORTFOLIOS["orta_riskli"]

        no_penalty = opt.optimize_portfolio(scores, sens, prev_weights=prev, turnover_penalty=0.0)
        with_penalty = opt.optimize_portfolio(scores, sens, prev_weights=prev, turnover_penalty=5.0)

        dist_no = sum(abs(no_penalty[a] - prev.get(a, 0)) for a in de.ASSETS)
        dist_with = sum(abs(with_penalty[a] - prev.get(a, 0)) for a in de.ASSETS)
        assert dist_with <= dist_no + 0.5   # ceza varken önceki ağırlığa daha yakın (ya da eşit)


class TestRunEngineV7:

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_produces_valid_allocation(self, profile):
        result = opt.run_engine_v7(profile)
        assert sum(result["final_alloc"].values()) == pytest.approx(100.0, abs=0.01)
        for asset, value in result["final_alloc"].items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01

    def test_crisis_mode_uses_same_override_as_v6(self):
        """Kriz modunda v7 de v6 ile AYNI CRISIS_OVERRIDE'ı kullanmalı (optimizer bypass)."""
        crisis_date = None
        for d in de.get_available_dates():
            df, _ = de.load_data(date=d)
            df = de.compute_derived(df)
            is_crisis, _ = de.check_crisis(df.iloc[-1])
            if is_crisis:
                crisis_date = d
                break
        if crisis_date is None:
            pytest.skip("Veri setinde kriz tarihi bulunamadı")

        r7 = opt.run_engine_v7("orta_riskli", date=crisis_date)
        assert r7["mode"] == "KRİZ"
        assert r7["final_alloc"] == de.CRISIS_OVERRIDE

    def test_deterministic(self):
        r1 = opt.run_engine_v7("orta_riskli", date="2024-06-30")
        r2 = opt.run_engine_v7("orta_riskli", date="2024-06-30")
        assert r1["final_alloc"] == r2["final_alloc"]
