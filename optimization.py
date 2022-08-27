import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from random import choice, uniform
from datetime import datetime

from augmentation import augmentationDataFrame
from utility import drawFeatures

# Список параметров, которые CatBoost должен преобразовать в векторные данные
categoricalFeatures = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY', 'DISEASE_PART', "CITY_TYPE"]

# Считывание исходных данных для обучения в Pandas DataFrame
train = pd.read_csv('train.csv', sep=';', index_col=None, dtype={'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str, 'PATIENT_ID_COUNT':int})

# дополняем данные обучающей выборки
augmentationDataFrame(train, withClassify = False)

# Отбираем нужные поля для обучения. Этот список понадобится в дальнейшем для отображения значимости параметров
X = train[features := ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'CITIZENS', 'DISEASE_PART', 'CITY_TYPE']]
# отбираем поле как целевое значение регрессии
y = train[['PATIENT_ID_COUNT']]

# разделяем тренировочную выборку на две части - обучающую и проверяющую обучение
# соотношение 1/10 подобрано опытным путём
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 500)

# формируем данные для обучения CatBoost
pool_train = Pool(X_train, y_train, cat_features = categoricalFeatures)
# формируем данные для контроля обучения CatBoost
pool_test = Pool(X_test, cat_features = categoricalFeatures)

# Указываем приемлемые варианты значений параметров CatBoost.
# Depth выше 10 отброшены, как вызывающие переобучение модели
params = \
{
    'depth': [None, 7, 8, 9, 10],
    'l2_leaf_reg': [None, 5, 10, 30, 50, 100],
    'border_count': [None, 5, 10, 20, 30, 50, 100, 200],
    'bagging_temperature': [None, 0.03, 0.09, 0.25, 0.75],
    'random_strength': [None, 0.1, 0.2, 0.5, 0.7, 0.8],
    'max_ctr_complexity': [None, 1, 2, 3, 4, 5],
    'n_estimators': [None, 50, 100, 200, 300],
    'min_data_in_leaf': [None, 1, 50, 100, 200, 300, 400],
    'max_bin': [None, 100, 200, 300, 400, 500],
}

# создать модель CatBoost, с параметрами params и запустить обучение
def getModel(pool_train, X_test, y_test, params):
    model = CatBoostRegressor(task_type="GPU", random_seed=500, verbose=200, **params)
    model.fit(pool_train, eval_set=(X_test, y_test), use_best_model=True)

    return model


# сначала создаём модель без дополнительных параметров
model = getModel(pool_train, X_test, y_test, {})

# предсказываем значения для проверочной части обучающей выборки
y_prediction = model.predict(pool_test)
# сравниваем с проверочной частью обучающей выборки, вычисляя функцию r-квадрат
globalBestScore = r2_score(y_test, y_prediction)
# это же значение равно значению без параметров
noneScore = globalBestScore

# R-квадрат качества обучения этой модели, в сравнении с проверочной частью
# это будет базовым значением качества. Далее будем искать такие параметры,
# которые дадут большее значение качества
print("Default score:", globalBestScore)

# сохраняем в файл график значимости параметров данных
drawFeatures(model, columns = features)

# запускаем случайную выборку параметров некое большое количество раз
# (например, 500)
for _ in range(0, 500):
    # создаём новый набор данных, с полями как у params,
    # но со значением, равным одному из случайно выбранных значений из списка
    currentParam = {}
    for copyName, _ in params.items():
        currentParam[copyName] = choice(params[copyName])

    # параметры border_count и max_bin конфликтуют между собой
    # если они оба не None, тогда один из них, выбранный случайным образом,
    # становится None
    if currentParam["border_count"] is not None and currentParam["max_bin"] is not None:
        if uniform(0, 1) == 0:
            currentParam["max_bin"] = None
        else:
            currentParam["border_count"] = None

    # запускаем обучение модели
    model = getModel(pool_train, X_test, y_test, currentParam)

    # получаем значение качества обучения
    y_prediction = model.predict(pool_test)
    score = r2_score(y_test, y_prediction)

    # если найдено более высокое значение качества
    if globalBestScore < score:
        globalBestScore = score

        # то выводим подобранные параметры и заодно сохраняем графики значимости параметров данных
        # более качественными признавались те, где меньше разница между значимостью разных параметров
        print("best score:", globalBestScore, "best parameters:", currentParam, "time:", datetime.now().strftime("%Y-%m-%d-%H-%M"))
        drawFeatures(model, features)

# Теперь ещё раз проверяем каждый параметр отдельно от других
for parameterName, parameterValues in params.items():
    bestValue = None
    bestScore = 0

    # создаём копию списка настроек
    currentParam = {}
    for copyName, _ in params.items():
        currentParam[copyName] = None

    # для каждого из списка значений настроек
    for parameterValue in parameterValues:
        # Значение None проверять нет смысла, оно проверялось в самом начале
        if parameterValue is None:
            continue

        # устанавливаем настройку
        currentParam[parameterName] = parameterValue

        # и выполняем обучение модели
        model = getModel(pool_train, X_test, y_test, currentParam)

        # проверяем качество обучения
        y_prediction = model.predict(pool_test)
        score = r2_score(y_test, y_prediction)

        # если это лучшее значение для параметра
        if bestScore < score:
            # обновляем значение оценки, сохраняем значение параметра
            bestScore = score
            bestValue = parameterValue

            # если значение оценки выше чем значение оценки без параметров, то выводим значение параметра
            if score > noneScore:
                print(parameterName, "best value:", bestValue, "best score:", bestScore, "time:", datetime.now().strftime("%Y-%m-%d-%H-%M"))
                drawFeatures(model, features)

        # если оценка даже лучше, чем за 500 предыдущих проверок
        # тогда выводим об этом радостное сообщение
        # (но такого не бывало, разумеется)
        if globalBestScore < score:
            globalBestScore = score
            print("Global best score!", bestScore)

