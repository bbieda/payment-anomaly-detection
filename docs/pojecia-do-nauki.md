# Pojęcia do nauki (w kolejności przydatności)

Przy każdym: po co, gdzie w pracy/kodzie, gdzie szukać (fraza do wyszukiwarki lub źródło z pracy).

## A. Dane i ewaluacja (najważniejsze – tu będą pytania)
1. **Niezbalansowanie klas (class imbalance)** – 0,17% oszustw; dlaczego accuracy jest bezużyteczne. [2], [1], [6]
2. **Macierz pomyłek: TP/FP/FN/TN, precision, recall, F1** – `anomaly/metrics.py`.
3. **Krzywa ROC i ROC AUC; krzywa PR i PR-AUC (average precision)** – różnica, baseline PR = odsetek oszustw. [6]
4. **Dobór progu decyzyjnego** (na walidacji, nie na teście), **koszt błędów** (FN drożej niż FP). `best_f1_threshold`.
5. **Podział czasowy vs losowy; wyciek danych (data leakage)** – dlaczego Twój podział różni się od [2]. [1]
6. **Walidacja krzyżowa (k-fold, stratyfikowana)** – bo [2] używa 5-fold; ty możesz odtworzyć ich ustawienie dla porównania.
7. **Bootstrap, przedział ufności, powtórzenia (seedy)** – analiza statystyczna (rozdz. 6.6).
8. **Dryf koncepcji (concept drift), opóźnienie weryfikacji (verification latency)** – [1]; w pracy jako ograniczenie.

## B. Modele (to, co porównują prace z literatury)
9. **Regresja logistyczna** (baseline), regularyzacja L2, `class_weight`.
10. **Drzewo decyzyjne** (kryterium Gini/entropia), przeuczenie, głębokość.
11. **Bagging i Random Forest** (losowanie próbek i cech, OOB) – Breiman 2001.
12. **Gradient boosting i XGBoost** (addytywny model, funkcja straty + regularyzacja, `scale_pos_weight`, `learning_rate`, `max_depth`) – [5].
13. **Isolation Forest** (długość ścieżki izolacji, wynik anomalii, parametr `contamination`) – [4].
14. **Local Outlier Factor** (k-sąsiedzi, lokalna gęstość, `novelty=True`) – Breunig 2000.
15. **One-Class SVM** (jądro RBF, `nu`) – występuje w [2].
16. **Autoenkoder / RBM / GAN w detekcji anomalii** (błąd rekonstrukcji jako wynik) – [2]; wystarczy zrozumieć ideę.
17. **Uczenie nadzorowane vs nienadzorowane vs półnadzorowane (novelty detection)** – dlaczego IF trenujesz bez etykiet, a etykiet używasz tylko do progu i oceny.
18. **Podejścia hybrydowe** (wynik nienadzorowany jako cecha dla klasyfikatora) – [3].

## C. Postępowanie z niezbalansowaniem
19. **Wagi klas, undersampling, oversampling, SMOTE** – tylko na zbiorze treningowym! [Chawla 2002]; [2] używa losowego undersamplingu.
20. **Kalibracja prawdopodobieństw** – wynik modelu ≠ prawdopodobieństwo oszustwa (szczególnie po resamplingu). [1]

## D. Dane i przetwarzanie
21. **PCA** (po co cechy V1–V28; dlaczego nie da się ich zinterpretować i zastosować do prawdziwych transakcji).
22. **Skalowanie cech** (RobustScaler, StandardScaler) i **Pipeline** w scikit-learn (unikanie wycieku).
23. **Cechy czasu cykliczne** (sin/cos godziny) – `anomaly/schemas.py`.

## E. Aplikacja
24. **REST, FastAPI, Pydantic, walidacja, kody HTTP, CORS, wątki (sync vs async w FastAPI)**.
25. **SQLAlchemy, SQL, migracje (Alembic)**, SQLite vs PostgreSQL.
26. **Serializacja modeli (joblib), wersjonowanie modeli**; bezpieczeństwo ładowania `pickle` (nie ładuj cudzych plików).
27. **React, komponenty, stan, polling, Recharts** (frontend).
28. **Open Banking / PSD2, AISP, OAuth2, zgoda użytkownika, polling vs webhook** – przed PoC z bankiem.

## Kolejność nauki (propozycja)
Tydzień 1: A1–A5, B9–B13 (to wystarczy do spotkania II). Tydzień 2: A6–A7, C19–C20, B14–B15. Później: reszta wg potrzeb pracy.
Tip: dla każdego modelu przeczytaj dokumentację scikit-learn/XGBoost (sekcja „User Guide”) i spróbuj w notebooku na danych syntetycznych (`anomaly/synthetic.py`).

## F. Nowe modele i analiza
28. **SVM i jądro RBF** (margines, wektory podparcia, parametr `gamma`) oraz **One-Class SVM** (`nu` jako górne ograniczenie odsetka outlierów i dolne wektorów podparcia).
29. **Autoenkoder i błąd rekonstrukcji** (koder, wąskie gardło, dekoder; dlaczego trening tylko na normalnych; wczesne zatrzymanie).
30. **Wrażliwość metod odległościowych i jądrowych na skalę cech** (długi ogon kwoty, `log1p`).
31. **Bootstrap parami** (te same próbki dla obu modeli, przedział percentylowy różnicy, ograniczenia: tylko zmienność testu, wielokrotne porównania).
