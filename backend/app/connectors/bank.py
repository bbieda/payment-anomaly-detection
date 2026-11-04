"""Szkielet konektora bankowego (Open Banking / PSD2) -- do zrobienia w późniejszej fazie.

Plan PoC: dostawca agregujący (np. Enable Banking, tryb „restricted production” dla własnych kont),
OAuth/zgoda użytkownika, cykliczne pobieranie (polling) transakcji i mapowanie na Transaction.
`from_psd2_item` mapuje typowy element listy transakcji w stylu Berlin Group NextGenPSD2
(pola do zweryfikowania z dokumentacją wybranego dostawcy).
"""
from datetime import datetime, timezone

from .base import Transaction, TransactionSource


def from_psd2_item(item: dict) -> Transaction:
    amt = item["transactionAmount"]
    ts = item.get("bookingDateTime") or item.get("bookingDate") or item.get("valueDate")
    return Transaction(
        id=str(item.get("transactionId") or item.get("entryReference")),
        timestamp=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) if "T" not in ts else datetime.fromisoformat(ts.replace("Z", "+00:00")),
        amount=abs(float(amt["amount"])), currency=amt.get("currency", "PLN"),
        merchant=item.get("creditorName") or item.get("debtorName"),
        category=item.get("merchantCategoryCode"))


class BankSource(TransactionSource):
    name = "bank"

    def __init__(self, provider: str = "enable_banking", credentials: dict | None = None):
        self.provider, self.credentials = provider, credentials

    def fetch(self, limit: int = 1) -> list[Transaction]:
        raise NotImplementedError("Konektor bankowy nie jest jeszcze zaimplementowany (zob. docs/bank_api.md).")
