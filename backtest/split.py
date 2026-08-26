from __future__ import annotations
"""
Walk-forward train/test tarih bölme yardımcıları.

Faz 1 (sinyal doğrulama) ve Faz 2 (ağırlık kalibrasyonu) bu modülü kullanarak
"SADECE o ana kadarki veriyle eğit, ondan sonrasını test et" kuralını uygular —
leakage'ı (gelecekteki veriyi kalibrasyona sızdırmayı) önlemenin tek yolu bu.

İki bölme stratejisi:
  expanding_window_splits : train penceresi her adımda BÜYÜR (baştan itibaren
                             tüm geçmiş kullanılır) — daha fazla veri, ama eski
                             rejimlerin ağırlığı zamanla "sulanır".
  rolling_window_splits   : train penceresi SABİT boyutta kayar — sadece son
                             N yılı görür, rejim değişimine daha çabuk uyum
                             sağlar ama daha az veriyle eğitilir.

Her ikisi de aynı `WalkForwardSplit` tipini üretir.
"""

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_dates: List[pd.Timestamp]
    test_dates: List[pd.Timestamp]

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]

    def __repr__(self) -> str:
        return (f"WalkForwardSplit(train={self.train_start.date()}..{self.train_end.date()} "
                f"[{len(self.train_dates)}g], test={self.test_start.date()}..{self.test_end.date()} "
                f"[{len(self.test_dates)}g])")


def _to_sorted_ts(dates) -> List[pd.Timestamp]:
    return sorted(pd.to_datetime(list(dates)))


def expanding_window_splits(dates, min_train_years: float = 2.0,
                             test_years: float = 1.0,
                             step_months: int = 6) -> Iterator[WalkForwardSplit]:
    """Train penceresi baştan itibaren her adımda genişler (expanding window)."""
    ts = _to_sorted_ts(dates)
    if not ts:
        return
    start = ts[0]
    min_train_end = start + pd.DateOffset(years=min_train_years)
    data_end = ts[-1]

    train_end = min_train_end
    while True:
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > data_end:
            break
        train_dates = [d for d in ts if start <= d <= train_end]
        test_dates = [d for d in ts if train_end < d <= test_end]
        if train_dates and test_dates:
            yield WalkForwardSplit(train_dates=train_dates, test_dates=test_dates)
        train_end = train_end + pd.DateOffset(months=step_months)


def rolling_window_splits(dates, train_years: float = 2.0,
                           test_years: float = 1.0,
                           step_months: int = 6) -> Iterator[WalkForwardSplit]:
    """Train penceresi SABİT boyutta kayar (rolling window)."""
    ts = _to_sorted_ts(dates)
    if not ts:
        return
    data_start = ts[0]
    data_end = ts[-1]

    train_start = data_start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > data_end:
            break
        train_dates = [d for d in ts if train_start <= d <= train_end]
        test_dates = [d for d in ts if train_end < d <= test_end]
        if train_dates and test_dates:
            yield WalkForwardSplit(train_dates=train_dates, test_dates=test_dates)
        train_start = train_start + pd.DateOffset(months=step_months)
