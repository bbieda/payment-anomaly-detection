# Eksperymenty (rozdział „Badania własne”)

```bash
# pełny przebieg na prawdziwych danych (kilkanaście–kilkadziesiąt minut, zależnie od komputera)
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all

# osobno: main | niu | imbalance | rolling
python -m anomaly.experiments --what main,rolling --seeds 5 --bootstrap 1000

# szybki test na danych syntetycznych (kilka sekund)
python -m anomaly.experiments --synthetic 20000 --fast --seeds 2 --bootstrap 100 --out /tmp/res
```

## Co robi każdy moduł
| `--what` | Co | Wyniki |
|---|---|---|
| `main` | 5 modeli, podział czasowy 70/15/15, próg z walidacji, `--seeds` powtórzeń, bootstrap CI na teście, opóźnienie predykcji | `tab_wyniki_glowne`, `tab_przedzialy`, krzywe PR/ROC, macierze pomyłek, wykres CI |
| `niu` | ustawienie zbliżone do [2]: losowa stratyfikowana CV (`--folds`), AUROC, niedopróbkowanie dla modeli nadzorowanych (`--undersample`) | `tab_niu`, `niu_auroc` |
| `imbalance` | brak / wagi klas / SMOTE / undersampling dla LogReg, RF, XGBoost (3 seedy) | `tab_niezbalansowanie`, `niezbalansowanie` |
| `rolling` | 4 okna czasowe (test = kolejne 15% danych, walidacja tuż przed) | `tab_okna`, `okna_pr_auc` |

Wyniki: `results/csv/*.csv`, `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/run_info.json` (seedy, parametry).

## Uczciwe zastrzeżenia (ujmij je w pracy)
- **Replikacja [2] jest przybliżona.** Znam z pracy: losowy podział, niedopróbkowanie dla modeli nadzorowanych, AUROC. Nie znam wszystkich szczegółów
  (stosunek klas po niedopróbkowaniu, hiperparametry), a modele RBM/GAN/autoenkoder nie są u mnie zaimplementowane. Piszesz więc „ustawienie zbliżone do [2]”, nie „replikacja”.
  Liczby z [2] w tabeli to tylko XGBoost i RF (AUROC 0,989 i 0,988).
- **Modele deterministyczne** (regresja logistyczna, w praktyce LOF z ustalonym ziarnem) mają zerowe odchylenie z seedów. To zmienność od treningu, nie od danych.
- **Bootstrap** uwzględnia zmienność składu zbioru testowego, ale przy kilkudziesięciu oszustwach przedziały będą szerokie. To wynik, który warto opisać, nie ukrywać.
- **Okna czasowe**: pierwsze okno ma mało danych treningowych (ok. 25%).
- Dobór hiperparametrów jest domyślny (bez strojenia). Strojenie (np. `RandomizedSearchCV` z `average_precision`) dodaj, jeśli zostanie czas.

## Użycie w LaTeX (Overleaf)
Skopiuj `results/tables/*.tex` do `tables/`, a `results/figures/*.pdf` do `figures/`, potem:
```latex
\input{tables/tab_wyniki_glowne.tex}
\begin{figure}[H]\centering
  \includegraphics[width=0.8\textwidth]{figures/krzywe_pr.pdf}
  \caption{Krzywe precision--recall modeli na zbiorze testowym}\label{fig:krzywe-pr}
\end{figure}
```
Tabele używają `booktabs` i `float` (opcja `[H]`), a liczby mają przecinek dziesiętny. Tabela `tab_niu` zawiera `\cite{niu2019}`.
