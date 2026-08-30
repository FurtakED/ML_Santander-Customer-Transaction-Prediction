# Santander Customer Transaction Prediction

**Соревнование:** [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction)
**Метрика:** ROC-AUC

## Результат

||Score|
|-|-|
|**Public Leaderboard**|0.89800|
|**Private Leaderboard**|0.89526|
|Локальный holdout (ансамбль)|0.9098|

Username на Kaggle: Лена ФУртак
Скриншот результата: [submission_screenshot.png](/submission_screenshot.png)

\---

## Задача

Бинарная классификация: предсказать, совершит ли клиент банка определённую транзакцию. Датасет анонимизирован — 200 числовых признаков (`var\\\_0` … `var\\\_199`) без описаний, сильный дисбаланс классов (\~90% / \~10%).

## Краткое описание подхода

### 1\. EDA

* Пропусков и дубликатов в данных нет
* Признаки практически не коррелируют друг с другом (|corr| < 0.01) и слабо коррелируют с таргетом (|corr| < 0.08) — указывает на отсутствие линейных зависимостей
* Распределения признаков train и test визуально совпадают, что подтверждено adversarial validation

### 2\. Baseline

Простые модели без feature engineering, на сырых 200 признаках:

|Модель|CV ROC-AUC|
|-|-|
|Logistic Regression|0.8590|
|LightGBM|0.8814|
|**Naive Bayes**|**0.8882**|

Naive Bayes неожиданно обошёл LightGBM на baseline-этапе это прямое следствие практической независимости признаков, обнаруженной в EDA (одна из предпосылок метода оказалась близка к реальности для этого датасета).

### 3\. Подбор гиперпараметров

`RandomizedSearchCV` (5-fold CV, ROC-AUC) для LightGBM, XGBoost, CatBoost:

|Модель|Baseline|После тюнинга|
|-|-|-|
|LightGBM|0.8814|0.8919|
|XGBoost|0.8677|0.8928|
|CatBoost|0.8752|0.8935|

### 4\. Feature Engineering (ключевой этап, наибольший прирост)

* **Row-статистики**: sum, mean, std, min, max, median, skew, kurtosis по каждой строке
* **Count-признаки**: частота каждого значения признака в train и test
* **Real/synthetic split теста**: около половины строк `test.csv` — синтетические (сгенерированы комбинированием значений настоящих строк). Настоящие строки почти всегда содержат хотя бы одно значение признака, уникальное в пределах test, синтетические — нет. Частотные признаки пересчитаны **только по train и настоящим строкам test**, что устранило искажение от синтетических данных

|Модель|До FE|После FE|
|-|-|-|
|LightGBM|0.8919|0.9046|
|CatBoost|0.8935|0.9052|
|XGBoost|0.8928|0.9089 (holdout)|

### 5\. Ансамбль

5 моделей (Logistic Regression, Naive Bayes, LightGBM, XGBoost, CatBoost) объединены через:

* Простое усреднение
* **Взвешенный блендинг** (веса подобраны оптимизацией ROC-AUC на out-of-fold предсказаниях)
* Стекинг (мета-модель — Logistic Regression)

|Метод|Holdout ROC-AUC|
|-|-|
|**Weighted Blend**|**0.9098**|
|Stacking|0.9095|
|XGBoost (single, best)|0.9089|

Веса взвешенного блендинга: XGBoost — 0.627, CatBoost — 0.214, Naive Bayes — 0.108, LightGBM — 0.028, Logistic Regression — 0.023. Несмотря на невысокий собственный score, Naive Bayes и Logistic Regression получили ненулевые веса — их ошибки слабо коррелируют с ошибками бустингов, что подтверждает пользу разнородных моделей в ансамбле.

## Что сработало лучше всего

* **Real/synthetic признак** это самый значимый источник прироста (+0.01–0.013 ROC-AUC), специфичный именно для этого датасета приём
* Взвешенный блендинг разнородных моделей дал дополнительный прирост (+0.003) над лучшей одиночной моделью
* Naive Bayes оказался неожиданно сильным baseline благодаря независимости признаков, обнаруженной в EDA

## Что не сработало

* Дальнейший тюнинг гиперпараметров после первого прохода `RandomizedSearchCV` давал прирост, измеряемый тысячными, а feature engineering оказался кратно эффективнее для улучшения метрики
* Метод определения real/synthetic строк — эвристический (основан на статистической редкости значений), не даёт 100% гарантии корректной классификации каждой отдельной строки, но подтверждён эмпирически через рост CV ROC-AUC

## Инструкции по воспроизведению результата

### 1\. Окружение

```bash
conda create -n ml\\\_project python=3.11 -y
conda activate ml\\\_project
conda install pandas numpy scikit-learn matplotlib seaborn jupyter lightgbm -y
pip install xgboost catboost psutil
```

Либо через `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2\. Данные

Скачать `train.csv`, `test.csv`, `sample\\\_submission.csv` со страницы соревнования (требуется принять правила):

```bash
kaggle competitions download -c santander-customer-transaction-prediction -p data/
```

Распаковать архив в папку `data/`.

### 3\. Запуск

Открыть `notebooks/santander\\\_solution.ipynb` и выполнить все ячейки по порядку (`Kernel → Restart \\\& Run All`).

### 

