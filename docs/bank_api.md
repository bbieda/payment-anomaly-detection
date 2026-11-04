# Wybór API bankowego do PoC (stan na X 2026; zweryfikuj przed decyzją)

Cel: pobieranie **własnych** transakcji do dashboardu. Dostęp do rachunków przez API w UE regulowany jest przez PSD2:
dane wydaje bank (ASPSP) tylko zarejestrowanemu dostawcy usług informacji o rachunku (AISP) za zgodą klienta.

| Opcja | Co ustaliłem | Ocena dla pracy |
|---|---|---|
| **Revolut Open Banking API (bezpośrednio)** | Produkcja wymaga certyfikatu eIDAS/OBIE, czyli bycia regulowanym TPP; inne podmioty mogą się zgłosić z zapytaniem o dostęp (dokumentacja Revolut Developer) | Nierealne dla osoby prywatnej |
| **Revolut Business API** | Integracja dla kont firmowych (komunikat Revolut z 2018 r.); nie dotyczy zwykłego konta osobistego | Tylko jeśli masz konto Business |
| **Agregator Enable Banking** | Darmowy tryb „restricted production”: dane tylko z kont, które sam połączysz, użytek indywidualny i niekomercyjny; ok. 2700 europejskich banków; konta trzeba wcześniej powiązać w panelu | **Najlepszy kandydat do PoC** (do sprawdzenia: czy Twój bank jest obsługiwany) |
| GoCardless Bank Account Data | Według przeglądu openbankingtracker.com nowe rejestracje są wyłączone | Odpada (sprawdź aktualny status) |
| Sandbox dowolnego dostawcy | Dane testowe, bez prawdziwych kont | Dobry pierwszy krok (budowa i testy konektora) |

## Proponowany plan PoC
1. Sprawdź w Enable Banking, czy Twój bank jest na liście; załóż aplikację w trybie sandbox.
2. Zaimplementuj `BankSource.fetch` (OAuth/zgoda, polling), mapowanie `from_psd2_item`, testy na zamockowanych odpowiedziach.
3. Dopiero potem tryb produkcyjny z własnym kontem (zgoda trwa ograniczony czas, wymaga odnawiania).
4. Dane to Twoje dane osobowe: nie commituj kluczy i transakcji do repo (`.env`, `.gitignore`), nie pokazuj ich w pracy bez anonimizacji.

## Uwaga metodyczna
Model Kaggle (V1–V28) nie oceni transakcji z banku. Do danych bankowych służy schemat `generic_tx` (model profilu użytkownika);
jego jakości nie da się policzyć jak PR-AUC bez etykiet, więc w pracy opisz go jako demonstrację/PoC, a właściwe porównanie robisz na Kaggle.
