from __future__ import annotations
"""
decision_engine.py için yapısal doğrulama testleri.

Amaç: motorun çıktısının HER ZAMAN doğru olan invaryantlarını (toplam=%100,
sınırlar, negatif olmama, bütçe-nötrlük vb.) garanti altına almak. Gerçek
piyasa verisine (portfolio_data/) bağımlı DEĞİL — hepsi sentetik/kontrollü
girdilerle çalışıyor, böylece pipeline hiç çalıştırılmamış bir makinede bile
`pytest` ile çalışır.

Çalıştır:  python -m pytest tests/ -v
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import decision_engine as de


ALL_PROFILES = list(de.BASE_PORTFOLIOS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────
def random_scores(seed: int) -> dict:
    rng = random.Random(seed)
    return {a: rng.uniform(0.0, 1.0) for a in de.ASSETS}


def extreme_scores(direction: str) -> dict:
    """Tüm varlıkları aynı yöne (max cazip / max cazip değil) iten uç senaryo."""
    return {a: (0.99 if direction == "up" else 0.01) for a in de.ASSETS}


# ─────────────────────────────────────────────────────────────────────────
# 1. score_to_adjustment — eşik / yön / duyarlılık
# ─────────────────────────────────────────────────────────────────────────
class TestScoreToAdjustment:

    def test_neutral_zone_gives_zero(self):
        for score in (0.5, 0.5 + de.NEUTRAL_THRESHOLD - 0.001, 0.5 - de.NEUTRAL_THRESHOLD + 0.001):
            assert de.score_to_adjustment(score, sensitivity=1.0) == 0.0

    def test_direction_matches_sign_of_deviation(self):
        assert de.score_to_adjustment(0.9, 1.0) > 0
        assert de.score_to_adjustment(0.1, 1.0) < 0

    def test_magnitude_increases_with_deviation(self):
        a = abs(de.score_to_adjustment(0.60, 1.0))   # hafif bölge
        b = abs(de.score_to_adjustment(0.75, 1.0))   # orta bölge
        c = abs(de.score_to_adjustment(0.95, 1.0))   # güçlü bölge
        assert a < b < c

    @pytest.mark.parametrize("sensitivity", [0.6, 1.0, 1.4])
    def test_sensitivity_scales_linearly(self, sensitivity):
        base = de.score_to_adjustment(0.9, 1.0)
        scaled = de.score_to_adjustment(0.9, sensitivity)
        assert scaled == pytest.approx(base * sensitivity)


# ─────────────────────────────────────────────────────────────────────────
# 2. apply_adjustments — FIX-7 / FIX-8 / FIX-9 invaryantları
# ─────────────────────────────────────────────────────────────────────────
class TestApplyAdjustments:

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    @pytest.mark.parametrize("seed", range(30))
    def test_sums_to_exactly_100(self, profile, seed):
        scores = random_scores(seed)
        sens = de.PROFILE_SENSITIVITY[profile]
        _, final = de.apply_adjustments(de.BASE_PORTFOLIOS[profile], scores, sens)
        assert sum(final.values()) == pytest.approx(100.0, abs=0.01)

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    @pytest.mark.parametrize("seed", range(30))
    def test_never_breaches_asset_bounds(self, profile, seed):
        scores = random_scores(seed)
        sens = de.PROFILE_SENSITIVITY[profile]
        _, final = de.apply_adjustments(de.BASE_PORTFOLIOS[profile], scores, sens)
        for asset, value in final.items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01, (
                f"{profile}/seed={seed}: {asset}={value} sınır dışı [{lo},{hi}]")

    @pytest.mark.parametrize("direction", ["up", "down"])
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_extreme_uniform_scores_still_respect_bounds_and_100(self, direction, profile):
        """Tüm skorlar aynı yönde uç değerdeyken (en zorlayıcı senaryo)."""
        scores = extreme_scores(direction)
        sens = de.PROFILE_SENSITIVITY[profile]
        _, final = de.apply_adjustments(de.BASE_PORTFOLIOS[profile], scores, sens)
        assert sum(final.values()) == pytest.approx(100.0, abs=0.01)
        for asset, value in final.items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01

    def test_no_negative_allocations(self):
        for profile in ALL_PROFILES:
            for seed in range(20):
                scores = random_scores(seed)
                _, final = de.apply_adjustments(
                    de.BASE_PORTFOLIOS[profile], scores, de.PROFILE_SENSITIVITY[profile])
                assert all(v >= 0 for v in final.values())

    def test_base_zero_asset_never_gets_negative_adjustment(self):
        """FIX-7: base=0 olan bir varlığa asla negatif ayar uygulanmaz."""
        for profile in ALL_PROFILES:
            base = de.BASE_PORTFOLIOS[profile]
            zero_assets = [a for a, v in base.items() if v == 0]
            if not zero_assets:
                continue
            scores = extreme_scores("down")   # hepsini en kötü skora zorla
            adj_map, _ = de.apply_adjustments(base, scores, de.PROFILE_SENSITIVITY[profile])
            for a in zero_assets:
                assert adj_map[a] >= 0, f"{profile}: {a} base=0 iken negatif ayar aldı"

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    @pytest.mark.parametrize("seed", range(20))
    def test_adjustments_are_budget_neutral(self, profile, seed):
        """FIX-8: pozitif ayarların toplamı negatiflerin toplamına (mutlak) eşit olmalı."""
        scores = random_scores(seed)
        adj_map, _ = de.apply_adjustments(
            de.BASE_PORTFOLIOS[profile], scores, de.PROFILE_SENSITIVITY[profile])
        pos = sum(v for v in adj_map.values() if v > 0)
        neg = sum(-v for v in adj_map.values() if v < 0)
        if pos > 0 and neg > 0:   # her iki yönde de ayar varsa denge şartı geçerli
            assert pos == pytest.approx(neg, abs=0.05)

    def test_neutral_asset_untouched_when_no_other_asset_hits_bound(self):
        """Bir varlığın skoru tam nötrse VE hiçbir varlık sınıra çarpmıyorsa,
        o varlık base değerinde kalmalı (komşularının ayarından etkilenmemeli)."""
        scores = {a: 0.5 for a in de.ASSETS}   # hepsi nötr → hiçbir ayar yok, hiçbir sınır tetiklenmez
        for profile in ALL_PROFILES:
            _, final = de.apply_adjustments(
                de.BASE_PORTFOLIOS[profile], scores, de.PROFILE_SENSITIVITY[profile])
            for asset, base_val in de.BASE_PORTFOLIOS[profile].items():
                assert final[asset] == pytest.approx(base_val, abs=0.01)

    def test_regression_known_bug_case(self):
        """
        Gerçek bir çalıştırmada yakalanan somut hata senaryosu: net +25pt talep
        (bazı varlıklar +5/+10, bazıları -5, ayarsız kalanlar dahil TÜM varlıklar
        eskiden tek bir 100/125 çarpanıyla küçültülüyordu). Bu artık olmamalı.
        """
        base = de.BASE_PORTFOLIOS["orta_riskli"]
        # skorları elle, orijinal hata senaryosundaki ayar büyüklüklerini üretecek şekilde kur
        scores = {
            "mevduat": 0.65, "doviz": 0.30, "altin": 0.30, "tahvil": 0.75,
            "yatirim_fonu": 0.75, "hisse": 0.75, "temettu_hisse": 0.5, "kripto": 0.5,
        }
        adj_map, final = de.apply_adjustments(base, scores, de.PROFILE_SENSITIVITY["orta_riskli"])
        assert sum(final.values()) == pytest.approx(100.0, abs=0.01)
        for asset, value in final.items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01


# ─────────────────────────────────────────────────────────────────────────
# 3. _normalize_within_bounds — doğrudan birim testler
# ─────────────────────────────────────────────────────────────────────────
class TestNormalizeWithinBounds:

    def test_simple_case_no_bound_conflict(self):
        values = {"a": 50, "b": 30}
        bounds = {"a": (0, 100), "b": (0, 100)}
        out = de._normalize_within_bounds(values, bounds, target=100.0)
        assert sum(out.values()) == pytest.approx(100.0)
        assert out["a"] / out["b"] == pytest.approx(50 / 30, rel=0.01)

    def test_bound_conflict_clips_and_redistributes(self):
        """b, tavanı 40 olan bir varlık; talep ettiği pay tavanı aşıyor —
        klip edilip fazlası a'ya (serbest varlık) gitmeli."""
        values = {"a": 20, "b": 80}
        bounds = {"a": (0, 100), "b": (0, 40)}
        out = de._normalize_within_bounds(values, bounds, target=100.0)
        assert out["b"] == pytest.approx(40.0, abs=0.01)
        assert out["a"] == pytest.approx(60.0, abs=0.01)
        assert sum(out.values()) == pytest.approx(100.0)

    def test_all_zero_with_remaining_budget_splits_evenly(self):
        values = {"a": 0, "b": 0}
        bounds = {"a": (0, 100), "b": (0, 100)}
        out = de._normalize_within_bounds(values, bounds, target=100.0)
        assert out["a"] == pytest.approx(50.0, abs=0.5)
        assert out["b"] == pytest.approx(50.0, abs=0.5)
        assert sum(out.values()) == pytest.approx(100.0)

    def test_never_exceeds_bounds_even_under_stress(self):
        rng = random.Random(42)
        for _ in range(200):
            n = rng.randint(2, 8)
            values, bounds = {}, {}
            for i in range(n):
                lo = rng.uniform(0, 20)
                hi = lo + rng.uniform(5, 40)
                bounds[i] = (lo, hi)
                values[i] = rng.uniform(0, 100)
            out = de._normalize_within_bounds(values, bounds, target=100.0)
            total_lo = sum(b[0] for b in bounds.values())
            total_hi = sum(b[1] for b in bounds.values())
            if total_lo <= 100 <= total_hi:   # sadece matematiksel olarak mümkünse kontrol et
                for k, v in out.items():
                    lo, hi = bounds[k]
                    assert lo - 0.1 <= v <= hi + 0.1


# ─────────────────────────────────────────────────────────────────────────
# 4. check_crisis — hard/soft tetikleyiciler
# ─────────────────────────────────────────────────────────────────────────
class TestCheckCrisis:

    def test_vix_hard_trigger(self):
        row = pd.Series({"risk_score": 0.3, "vix_level": 45.0, "cds_level": 200.0})
        is_crisis, reason = de.check_crisis(row)
        assert is_crisis and "VIX" in reason

    def test_cds_hard_trigger(self):
        row = pd.Series({"risk_score": 0.3, "vix_level": 15.0, "cds_level": 750.0})
        is_crisis, reason = de.check_crisis(row)
        assert is_crisis and "CDS" in reason

    def test_soft_risk_score_trigger(self):
        row = pd.Series({"risk_score": 0.85, "vix_level": 15.0, "cds_level": 200.0})
        is_crisis, reason = de.check_crisis(row)
        assert is_crisis and "SOFT" in reason

    def test_no_trigger_under_thresholds(self):
        row = pd.Series({"risk_score": 0.5, "vix_level": 20.0, "cds_level": 300.0})
        is_crisis, _ = de.check_crisis(row)
        assert not is_crisis

    def test_missing_cds_does_not_crash_or_false_trigger(self):
        """cds_level=None (bkz. turkey_cds_5y kaynağı yoksa) kriz tetiklememeli."""
        row = pd.Series({"risk_score": 0.4, "vix_level": 15.0, "cds_level": None})
        is_crisis, _ = de.check_crisis(row)
        assert not is_crisis

    def test_crisis_override_sums_to_100(self):
        assert sum(de.CRISIS_OVERRIDE.values()) == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────
# 5. Yapılandırma tutarlılığı (config-level sağlamlık)
# ─────────────────────────────────────────────────────────────────────────
class TestConfigSanity:

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_base_portfolios_sum_to_100(self, profile):
        assert sum(de.BASE_PORTFOLIOS[profile].values()) == pytest.approx(100.0)

    def test_asset_bounds_feasible_for_target_100(self):
        """Sınır tablosunun kendisi %100'e ulaşmayı imkansız kılmamalı."""
        total_lo = sum(lo for lo, hi in de.ASSET_BOUNDS.values())
        total_hi = sum(hi for lo, hi in de.ASSET_BOUNDS.values())
        assert total_lo <= 100.0 <= total_hi

    def test_base_portfolios_within_their_own_bounds(self):
        """Başlangıç (base) portföyler bile tanımlı sınırların dışında olmamalı."""
        for profile, alloc in de.BASE_PORTFOLIOS.items():
            for asset, value in alloc.items():
                lo, hi = de.ASSET_BOUNDS[asset]
                assert lo <= value <= hi, f"{profile}/{asset}={value} taban sınır dışı [{lo},{hi}]"

    def test_all_assets_have_bounds_and_labels(self):
        for a in de.ASSETS:
            assert a in de.ASSET_BOUNDS
            assert a in de.ASSET_LABELS

    def test_profile_sensitivity_ordering(self):
        """Riskli profil, az riskliden daha duyarlı olmalı (motorun temel varsayımı)."""
        assert (de.PROFILE_SENSITIVITY["az_riskli"]
                < de.PROFILE_SENSITIVITY["orta_riskli"]
                < de.PROFILE_SENSITIVITY["cok_riskli"])


# ─────────────────────────────────────────────────────────────────────────
# 6. rolling_percentile_rank — sınır davranışları
# ─────────────────────────────────────────────────────────────────────────
class TestRollingPercentileRank:

    def test_value_at_series_max_is_near_one(self):
        s = pd.Series(range(100))
        assert de.rolling_percentile_rank(s, 99) == pytest.approx(1.0)

    def test_value_at_series_min_is_near_zero(self):
        s = pd.Series(range(100))
        assert de.rolling_percentile_rank(s, 0) == pytest.approx(0.01, abs=0.02)

    def test_empty_series_returns_neutral(self):
        s = pd.Series([], dtype=float)
        assert de.rolling_percentile_rank(s, 5.0) == 0.5

    def test_small_series_falls_back_to_full_history(self):
        s = pd.Series([1, 2, 3, 4, 5])   # < 30 nokta -> fallback
        assert 0.0 <= de.rolling_percentile_rank(s, 3) <= 1.0

    def test_only_uses_trailing_window(self):
        # İlk 300 gün hep 0, son 252 gün hep 100 -> pencere sadece son kısmı görmeli
        s = pd.Series([0] * 300 + [100] * 252)
        # 50 değeri, sadece [100]'lerden oluşan pencerede minimumun bile altında olmalı
        r = de.rolling_percentile_rank(s, 50, window=252)
        assert r == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────
# 7. Uçtan uca (gerçek veri varsa) — yoksa atlanır
# ─────────────────────────────────────────────────────────────────────────
DATA_AVAILABLE = de.MERGED_CSV.exists()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="portfolio_data/processed/merged_aligned_daily.csv yok — önce pipeline çalıştırılmalı")
class TestRunEngineIntegration:

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_run_engine_produces_valid_allocation(self, profile):
        result = de.run_engine(profile)
        assert sum(result["final_alloc"].values()) == pytest.approx(100.0, abs=0.01)
        for asset, value in result["final_alloc"].items():
            lo, hi = de.ASSET_BOUNDS[asset]
            assert lo - 0.01 <= value <= hi + 0.01

    def test_tl_alloc_matches_amount(self):
        result = de.run_engine("orta_riskli", amount=250_000)
        assert sum(result["tl_alloc"].values()) == pytest.approx(250_000, rel=0.001)

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_backtest_over_full_history_never_breaches_bounds(self, profile):
        dates = de.get_available_dates()
        sample = dates[::15]   # yaklaşık her 15 günde bir örnek, tam tarihçeyi tara
        for d in sample:
            result = de.run_engine(profile, date=d)
            total = sum(result["final_alloc"].values())
            assert total == pytest.approx(100.0, abs=0.05), f"{profile}/{d}: toplam={total}"
            for asset, value in result["final_alloc"].items():
                lo, hi = de.ASSET_BOUNDS[asset]
                assert lo - 0.05 <= value <= hi + 0.05, f"{profile}/{d}: {asset}={value}"

    def test_run_engine_is_deterministic(self):
        r1 = de.run_engine("orta_riskli", date="2024-06-30")
        r2 = de.run_engine("orta_riskli", date="2024-06-30")
        assert r1["final_alloc"] == r2["final_alloc"]
        assert r1["risk_score"] == r2["risk_score"]
