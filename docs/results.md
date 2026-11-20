# Wyniki eksperymentów

Zbiór: Kaggle ULB Credit Card Fraud (284 807 transakcji, 492 oszustwa). Podział czasowy 70/15/15: trening 199 364 (384 oszustwa), walidacja 42 721 (56), test 42 722 (52).
Próg decyzyjny każdego modelu: maksymalizacja F1 na walidacji. Główna miara: PR-AUC (pole pod krzywą precision–recall), bo przy 0,17% oszustw ROC AUC maskuje trudność zadania.
Wyniki generuje `python -m anomaly.experiments` (5 seedów: 42–46, bootstrap 500). Hiperparametry OCSVM i autoenkodera wg [2], reszta modeli ma własne wartości domyślne.

## 1. Wyniki główne (test, średnia ± odchylenie z 5 seedów)
| Model | PR-AUC | PR-AUC, 95% CI (seed 42) | ROC AUC | F1 | Precision | Recall | Opóźnienie [ms] |
|---|---|---|---|---|---|---|---|
| Regresja logistyczna | 0,711 ± 0,000 | 0,711 [0,578; 0,837] | 0,977 ± 0,000 | 0,759 | 0,943 | 0,635 | 1,8 |
| Random Forest | 0,770 ± 0,006 | 0,769 [0,654; 0,867] | 0,929 ± 0,006 | 0,824 | 0,923 | 0,746 | 27,4 |
| XGBoost | 0,763 ± 0,001 | 0,764 [0,652; 0,860] | 0,970 ± 0,005 | 0,805 | 0,875 | 0,746 | 2,3 |
| Isolation Forest | 0,047 ± 0,006 | 0,039 [0,025; 0,062] | 0,937 ± 0,005 | 0,119 | 0,067 | 0,504 | 11,8 |
| LOF | 0,002 ± 0,000 | 0,002 [0,001; 0,004] | 0,503 ± 0,030 | 0,010 | 0,005 | 0,058 | 2,6 |
| One-Class SVM | 0,038 ± 0,000 | 0,038 [0,025; 0,056] | 0,930 ± 0,000 | 0,102 | 0,056 | 0,538 | 2,1 |
| Autoenkoder | 0,235 ± 0,012 | 0,253 [0,167; 0,378] | 0,949 ± 0,008 | 0,396 | 0,337 | 0,500 | 1,7 |

Uwaga: przedziały ufności pochodzą z bootstrapu zbioru testowego dla seeda 42, więc wartość punktowa w nawiasie różni się od średniej z seedów (np. Isolation Forest 0,039 vs 0,047, autoenkoder 0,253 vs 0,235). Test ma tylko 52 oszustwa, dlatego przedziały są szerokie.

## 2. Różnice między modelami (bootstrap parami, PR-AUC, seed 42)
W każdej próbie bootstrapowej losowane są te same transakcje dla obu modeli. Różnica = PR-AUC(A) − PR-AUC(B). Uwzględniona jest tylko zmienność zbioru testowego, bez korekty na wielokrotne porównania (21 par).

| A | B | Różnica | 95% przedział | Istotna |
|---|---|---|---|---|
| Regresja logistyczna | Random Forest | -0,058 | [-0,144; 0,011] | nie |
| Regresja logistyczna | XGBoost | -0,053 | [-0,144; 0,017] | nie |
| Regresja logistyczna | Isolation Forest | 0,672 | [0,543; 0,788] | tak |
| Regresja logistyczna | LOF | 0,709 | [0,573; 0,839] | tak |
| Regresja logistyczna | One-Class SVM | 0,673 | [0,545; 0,795] | tak |
| Regresja logistyczna | Autoenkoder | 0,458 | [0,361; 0,554] | tak |
| Random Forest | XGBoost | 0,004 | [-0,017; 0,026] | nie |
| Random Forest | Isolation Forest | 0,730 | [0,624; 0,816] | tak |
| Random Forest | LOF | 0,767 | [0,653; 0,865] | tak |
| Random Forest | One-Class SVM | 0,731 | [0,626; 0,820] | tak |
| Random Forest | Autoenkoder | 0,516 | [0,411; 0,599] | tak |
| XGBoost | Isolation Forest | 0,725 | [0,618; 0,813] | tak |
| XGBoost | LOF | 0,763 | [0,645; 0,864] | tak |
| XGBoost | One-Class SVM | 0,726 | [0,619; 0,817] | tak |
| XGBoost | Autoenkoder | 0,511 | [0,415; 0,591] | tak |
| Isolation Forest | LOF | 0,038 | [0,023; 0,059] | tak |
| Isolation Forest | One-Class SVM | 0,001 | [-0,005; 0,012] | nie |
| Isolation Forest | Autoenkoder | -0,214 | [-0,317; -0,138] | tak |
| LOF | One-Class SVM | -0,036 | [-0,053; -0,023] | tak |
| LOF | Autoenkoder | -0,252 | [-0,371; -0,164] | tak |
| One-Class SVM | Autoenkoder | -0,215 | [-0,323; -0,139] | tak |

Istotnych jest 17 z 21 par. Nieistotne: regresja logistyczna vs Random Forest, regresja logistyczna vs XGBoost, Random Forest vs XGBoost oraz Isolation Forest vs One-Class SVM.
Modele nadzorowane są więc nieodróżnialne między sobą, ale wszystkie wyraźnie lepsze od wszystkich nienadzorowanych. Najlepszy z nienadzorowanych to autoenkoder.

## 3. Ustawienie zbliżone do [2] (losowa 5-krotna walidacja krzyżowa, AUROC)
| Model | Nasz AUROC (średnia ± odch.) | [2] |
|---|---|---|
| Regresja logistyczna | 0,977 ± 0,007 | n/d |
| Random Forest | 0,978 ± 0,007 | 0,988 |
| XGBoost | 0,979 ± 0,010 | 0,989 |
| Isolation Forest | 0,950 ± 0,010 | n/d |
| LOF | 0,570 ± 0,042 | n/d |
| One-Class SVM | 0,941 ± 0,012 | ok. 0,90 |
| Autoenkoder | 0,944 ± 0,010 | „powyżej 0,95” |

Modele nadzorowane wychodzą o ok. 0,01 niżej niż w [2], OCSVM wyżej. Autoenkoder jest nieco niżej, ale to uproszczona implementacja (`MLPRegressor`).

## 4. Niezbalansowanie klas (PR-AUC, test, podział czasowy)
| Model | brak | wagi klas | SMOTE | undersampling |
|---|---|---|---|---|
| Regresja logistyczna | 0,650 | 0,711 | 0,650 | 0,671 |
| Random Forest | 0,749 | 0,769 | 0,777 | 0,752 |
| XGBoost | 0,711 | 0,763 | 0,769 | 0,747 |

Różnice nie były testowane statystycznie. Dla RF i XGBoost wagi klas i SMOTE dają zbliżone wyniki; dla regresji logistycznej wagi są wyraźnie lepsze od SMOTE.

## 5. Strojenie modeli nienadzorowanych (`--what sweep`, jeden seed, wybór wg PR-AUC na walidacji)
| Model | Konfiguracja | Domyślna? | Wybrana? | PR-AUC walid. | PR-AUC test | ROC AUC test |
|---|---|---|---|---|---|---|
| LOF | n_neighbors=20, log_amount=False | tak |  | 0,004 | 0,002 | 0,500 |
| LOF | n_neighbors=50, log_amount=False |  |  | 0,056 | 0,044 | 0,824 |
| LOF | n_neighbors=100, log_amount=False |  | tak | 0,253 | 0,351 | 0,925 |
| LOF | n_neighbors=20, log_amount=True |  |  | 0,004 | 0,002 | 0,536 |
| LOF | n_neighbors=50, log_amount=True |  |  | 0,064 | 0,057 | 0,806 |
| LOF | n_neighbors=100, log_amount=True |  |  | 0,251 | 0,368 | 0,931 |
| One-Class SVM | nu=0.01, gamma=0.001, log_amount=False |  |  | 0,031 | 0,045 | 0,826 |
| One-Class SVM | nu=0.01, gamma=0.01, log_amount=False |  |  | 0,044 | 0,075 | 0,878 |
| One-Class SVM | nu=0.01, gamma=0.1, log_amount=False |  |  | 0,048 | 0,084 | 0,948 |
| One-Class SVM | nu=0.1, gamma=0.001, log_amount=False | tak |  | 0,027 | 0,038 | 0,930 |
| One-Class SVM | nu=0.1, gamma=0.01, log_amount=False |  |  | 0,033 | 0,056 | 0,937 |
| One-Class SVM | nu=0.1, gamma=0.1, log_amount=False |  |  | 0,050 | 0,090 | 0,950 |
| One-Class SVM | nu=0.01, gamma=0.001, log_amount=True |  |  | 0,044 | 0,062 | 0,885 |
| One-Class SVM | nu=0.01, gamma=0.01, log_amount=True |  |  | 0,052 | 0,085 | 0,886 |
| One-Class SVM | nu=0.01, gamma=0.1, log_amount=True |  |  | 0,053 | 0,092 | 0,945 |
| One-Class SVM | nu=0.1, gamma=0.001, log_amount=True |  |  | 0,038 | 0,057 | 0,934 |
| One-Class SVM | nu=0.1, gamma=0.01, log_amount=True |  |  | 0,041 | 0,068 | 0,935 |
| One-Class SVM | nu=0.1, gamma=0.1, log_amount=True |  | tak | 0,056 | 0,099 | 0,947 |
| Autoenkoder | hidden_layer_sizes=[16, 32, 32, 16], log_amount=False | tak |  | 0,146 | 0,253 | 0,934 |
| Autoenkoder | hidden_layer_sizes=[14, 7, 14], log_amount=False |  | tak | 0,199 | 0,336 | 0,956 |
| Autoenkoder | hidden_layer_sizes=[16, 32, 32, 16], log_amount=True |  |  | 0,140 | 0,248 | 0,944 |
| Autoenkoder | hidden_layer_sizes=[14, 7, 14], log_amount=True |  |  | 0,107 | 0,241 | 0,916 |

- **LOF:** o wyniku decyduje liczba sąsiadów, a nie skala kwoty. ROC AUC rośnie z 0,50 (k=20) przez 0,82 (k=50) do 0,93 (k=100), a `log_amount` prawie nic nie zmienia. Hipoteza o dominacji kwoty (`Amount`) nie została potwierdzona. Przypuszczenie (niezweryfikowane): LOF uczy się na danych zawierających oszustwa, więc przy małym k oszustwa tworzące zbite grupy wyglądają lokalnie normalnie.
- **Najlepsze konfiguracje leżą na brzegu siatki** (LOF k=100, OCSVM gamma=0,1), więc optimum może być dalej.
- Konfiguracje wybrane na walidacji są też najlepsze na teście, ale walidacja ma 56 oszustw, więc wybór jest niepewny. Różnic między modelami po strojeniu nie testowano statystycznie.
- `log_amount` pomaga OCSVM nieznacznie, nie pomaga LOF i szkodzi autoenkoderowi 14-7-14.

## 6. Stabilność w czasie (4 okna, seed 42, PR-AUC)
Test w oknie to kolejne 15% danych, walidacja tuż przed nim. Okno 4 to zbiór testowy z sekcji 1.

| Model | Okno 1 | Okno 2 | Okno 3 | Okno 4 |
|---|---|---|---|---|
| Regresja logistyczna | 0,750 | 0,603 | 0,856 | 0,711 |
| Random Forest | 0,799 | 0,689 | 0,859 | 0,769 |
| XGBoost | 0,805 | 0,686 | 0,855 | 0,764 |
| Isolation Forest | 0,435 | 0,016 | 0,031 | 0,039 |
| LOF | 0,003 | 0,001 | 0,003 | 0,002 |
| One-Class SVM | 0,300 | 0,018 | 0,026 | 0,038 |
| Autoenkoder | 0,705 | 0,096 | 0,126 | 0,253 |
| *Oszustw w teście* | 109 | 34 | 56 | 52 |

- Modele nadzorowane są stabilne (PR-AUC 0,60–0,86), bez jednego stałego zwycięzcy. Brak monotonicznego spadku, więc nie ma dowodu na dryf koncepcji (zbiór obejmuje tylko dwa dni).
- Modele nienadzorowane są bardzo niestabilne: Isolation Forest 0,435 w oknie 1 i 0,02–0,04 w pozostałych, autoenkoder od 0,096 do 0,705, OCSVM od 0,018 do 0,300. W oknie 2 Isolation Forest i OCSVM nie wykrywają żadnego oszustwa przy progu z walidacji (recall 0), mimo ROC AUC 0,95.
- Część zmienności wynika z odsetka oszustw w oknie (ok. 0,25%, 0,08%, 0,13% i 0,12% dla okien 1–4), bo PR-AUC zależy od częstości klasy. To wyjaśnienie wynika z arytmetyki, ale nie było osobno testowane.
- LOF ma ROC AUC poniżej 0,5 w oknach 1–3 (0,40, 0,31, 0,45), czyli oszustwa dostają niższe wyniki niż normalne transakcje. Zgadza się to z przypuszczeniem o maskowaniu przy małym k, ale nie było sprawdzane w innych oknach.

## Wnioski i ograniczenia
- Przy dostępnych etykietach modele nadzorowane (RF, XGBoost, regresja logistyczna) są wyraźnie lepsze od nienadzorowanych i nieodróżnialne między sobą.
- ROC AUC bliskie 0,93–0,95 przy PR-AUC rzędu 0,04 pokazuje, że ROC AUC nie nadaje się do oceny tego zadania.
- Wyniki opierają się na 52 oszustwach w teście, jednym zbiorze danych i cechach po PCA. Nie przenoszą się bezpośrednio na prawdziwe transakcje bankowe.
