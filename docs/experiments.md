# Eksperymenty (rozdział „Badania własne”)

```bash
# pełny przebieg na prawdziwych danych (kilkanaście–kilkadziesiąt minut, zależnie od komputera)
python -m anomaly.experiments --data data/raw/creditcard.csv --out results --what all

# osobno: main | niu | imbalance | rolling | pairs (pairs wczytuje results/csv/scores_test.npz, bez ponownego uczenia)
python -m anomaly.experiments --what main,rolling --seeds 5 --bootstrap 1000

# szybki test na danych syntetycznych (kilka sekund)
python -m anomaly.experiments --synthetic 20000 --fast --seeds 2 --bootstrap 100 --out /tmp/res
```

## Co robi każdy moduł
| `--what` | Co | Wyniki |
|---|---|---|
| `main` | 7 modeli (LogReg, RF, XGBoost, IF, LOF, OCSVM, autoenkoder), podział czasowy 70/15/15, próg z walidacji, `--seeds` powtórzeń, bootstrap CI na teście, opóźnienie predykcji | `tab_wyniki_glowne`, `tab_przedzialy`, krzywe PR/ROC, macierze pomyłek, wykres CI |
| `niu` | ustawienie zbliżone do [2]: losowa stratyfikowana CV (`--folds`), AUROC, niedopróbkowanie dla modeli nadzorowanych (`--undersample`) | `tab_niu`, `niu_auroc` |
| `imbalance` | brak / wagi klas / SMOTE / undersampling dla modeli nadzorowanych: LogReg, RF, XGBoost (3 seedy) | `tab_niezbalansowanie`, `niezbalansowanie` |
| `pairs` | bootstrap parami różnic PR-AUC dla wszystkich par modeli (te same transakcje w każdej próbie), liczony też automatycznie razem z `main` | `pary.csv`, `tab_pary`, `pary_roznice` |
| `rolling` | 4 okna czasowe (test = kolejne 15% danych, walidacja tuż przed) | `tab_okna`, `okna_pr_auc` |

Wyniki: `results/csv/*.csv`, `results/tables/*.tex`, `results/figures/*.pdf|png`, `results/run_info.json` (seedy, parametry).

## Uczciwe zastrzeżenia (ujmij je w pracy)
- **Replikacja [2] jest przybliżona.** Znam z pracy: losowy podział, niedopróbkowanie dla modeli nadzorowanych, AUROC. Nie znam wszystkich szczegółów
  (stosunek klas po niedopróbkowaniu, hiperparametry), a modele RBM, GAN, KNN i liniowy SVM nie są u mnie zaimplementowane. Piszesz więc „ustawienie zbliżone do [2]”, nie „replikacja”.
  Liczby z [2] w tabeli: XGBoost 0,989, RF 0,988, OCSVM ok. 0,90 (wartości z tekstu pracy; dla autoenkodera tekst podaje tylko „powyżej 0,95”).
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

## Nowe modele: One-Class SVM i autoenkoder
- **Hiperparametry wg [2]**: OCSVM `nu=0.1`, `gamma=0.001` (jądro RBF);
  autoenkoder: koder 16-32, dekoder 32-16 (ReLU). Pozostałe parametry (np. optymalizator Adam, wczesne zatrzymanie) to moje założenia.
- **Autoenkoder uczy się tylko na transakcjach normalnych** (wymuszone w rejestrze, niezależnie od `--normal-only`); wynik to błąd rekonstrukcji (MSE na cechach po skalowaniu).
  Zaimplementowany na `MLPRegressor`, więc to uproszczona wersja względem sieci z [2].
- **OCSVM uczy się na losowej próbce 20 tys. transakcji** (koszt rośnie szybciej niż liniowo); w odróżnieniu od autoenkodera widzi też oszustwa z tej próbki (jak IF i LOF).
- **Skalowanie:** jak w [2], `RobustScaler` na `Amount`. OCSVM i autoenkoder są wrażliwe na skalę cech (długi ogon kwoty). Opcjonalnie `log_amount=true`
  w parametrach (`log1p(Amount)` + standaryzacja) pozwala to sprawdzić. Domyślnie wyłączone.
- Nowe modele **nie były jeszcze uruchomione na prawdziwych danych**; testy i smoke-test działają na danych syntetycznych.

## Bootstrap parami
Dla pary modeli A i B w każdej próbie bootstrapowej losujemy te same transakcje testowe i liczymy PR-AUC(A) − PR-AUC(B). Raportujemy różnicę na pełnym teście, 95% przedział
percentylowy i orientacyjne $p$ (dwustronne, z rozkładu bootstrapowego). Różnica jest „istotna”, jeśli przedział nie zawiera zera. Uwaga: to zmienność zbioru testowego
(przy 52 oszustwach duża), a nie zmienność treningu; przy 28 parach nie ma korekty na wielokrotne porównania, więc traktuj wyniki jako opisowe.
