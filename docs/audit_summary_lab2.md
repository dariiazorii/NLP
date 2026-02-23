# Audit Summary Lab 2: Cleaning & Normalization

##  Статистика фільтрації
* **Початкова кількість рядків:** 5000
* **Видалено занадто коротких (< 5 слів):** 1590
* **Видалено сміття (без літер/лише цифри):** 1
* **Видалено дублікатів після маскування:** 395
* **ФІНАЛЬНА КІЛЬКІСТЬ (df_clean):** 3015

##  Маскування PII
* **<EMAIL>:** 0
* **<PHONE>:** 11
* **<URL>:** 12

##  Технічні деталі
* **Idempotence test:** Passed (Tags protected from homoglyph replacement)
* **Sentence split:** RegEx based, Ukrainian abbreviations aware.
