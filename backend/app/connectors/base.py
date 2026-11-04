"""Wspólny kontrakt dla źródeł transakcji. Nowe źródło = nowa klasa dziedzicząca z TransactionSource.

Całość systemu (scoring, baza, dashboard) zna tylko model Transaction -- nie wie, skąd dane pochodzą.
"""
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel


class Transaction(BaseModel):
    id: str
    timestamp: datetime
    amount: float
    currency: str = "PLN"
    merchant: str | None = None
    category: str | None = None
    features: dict[str, float] | None = None   # gotowe cechy (np. V1..V28 zbioru Kaggle)
    label: int | None = None                   # 1 = oszustwo, jeśli znane (CSV, symulacja)


class TransactionSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, limit: int = 1) -> list[Transaction]:
        """Zwraca kolejne transakcje ze źródła."""
