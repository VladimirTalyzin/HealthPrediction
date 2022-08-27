from pandas import read_csv
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from augmentation import augmentationDataFrame, dropAugmentation
from utility import drawFeatures

#additionalParams = {}
# параметры, собранные случайным поиском оптимизации
# указана дата нахождения и максимальный score на leaderboard, что они позволили получить
# 0.9223
additionalParams = {"depth": 10, 'bagging_temperature': 0.75, 'random_strength': 0.7, 'min_data_in_leaf': 50,  'max_bin': 400}

# Learning rate set to 0.141064
# time: 2022-08-20-03-46
# 0.9103
#additionalParams = {"depth": 10, 'bagging_temperature': 0.25, 'random_strength': 0.1, 'max_ctr_complexity': 5, 'min_data_in_leaf': 1, 'max_bin': 100}

# Learning rate set to 0.141064
# time: 2022-08-20-04-18
# 0.88
#additionalParams = {"depth": 10, 'random_strength': 0.8, 'max_ctr_complexity': 2, 'min_data_in_leaf': 100, 'max_bin': 400}

# Learning rate set to 0.141064
# time: 2022-08-19-23-45
# 0.89
#additionalParams = {"depth": 8, 'bagging_temperature': 0.25, 'random_strength': 0.1, 'max_ctr_complexity': 2, 'min_data_in_leaf': 200, 'max_bin': 200}

# Список параметров, которые CatBoost должен преобразовать в векторные данные
categoricalFeatures = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY', 'DISEASE_PART', "CITY_TYPE"]

# Считывание исходных данных для обучения и предсказания в Pandas DataFrame
train = read_csv('train.csv', sep=';', index_col = None, dtype = {'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str, 'PATIENT_ID_COUNT':int})
test = read_csv('test.csv', sep=';', index_col = None, dtype = {'PATIENT_SEX':str, 'MKB_CODE':str, 'ADRES':str, 'VISIT_MONTH_YEAR':str, 'AGE_CATEGORY':str})

# дополняем данные обучающей и тестовой выборки
augmentationDataFrame(train, withClassify = False)
augmentationDataFrame(test,  withClassify = False)

# отбираем нужные поля для обучения
X = train[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY', 'CITIZENS', 'DISEASE_PART', 'CITY_TYPE']]
# отбираем поле как целевое значение регрессии
y = train[['PATIENT_ID_COUNT']]

# разделяем тренировочную выборку на две части - обучающую и проверяющую обучение
# соотношение 1/10 подобрано опытным путём
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 500)

# формируем данные для обучения CatBoost
pool_train = Pool(X_train, y_train, cat_features = categoricalFeatures)
# формируем данные для контроля обучения CatBoost
pool_test = Pool(X_test, cat_features = categoricalFeatures)

# Создаём модель регрессии CatBoost с подобранными параметрами в additionalParams
# task_type: выполнение на GPU
# random_seed: заранее заданный SEED случайного поиска, чтобы получать идентичные результаты при разных обучениях
# verbose: выдаём информацию не о каждом шаге, а только о каждом 200 шаге. Также каждые 200 раз выясняется лучшая модель и выбирается как результат обучения
# learning_rate: подобрано опытным путём
# iterations: подобрано опытным путём. Хуже переобученной модели может быть только недоученная
model = CatBoostRegressor(task_type = 'GPU', random_seed = 500, verbose = 200, learning_rate = 0.14, iterations = 7000, **additionalParams)

# выполняем обучение
# pool_train: данные для обучения
# eval_set: данные для контроля обучения
# use_best_model: на каждом раунде из 200 обучений, выбираем лучшую модель и на следующем этапе далее обучаем её
# early_stopping_rounds: если лучшая модель найдена менее чем через 80 обучений из 200, а далее было
# только хуже, то значит далее модель будет переобучаться и обучение останавливается
model.fit(pool_train, eval_set = (X_test, y_test), use_best_model = True, early_stopping_rounds = 120)

# предсказываем данные для проверяющей части обучающей выборки
y_prediction = model.predict(pool_test)
# чтобы сравнить результат через функцию r-квадрат и получить оценку качества обучения
print("Score: ", r2_score(y_test, y_prediction))

# нарисовать и сохранить в файл относительные значения параметров обученной модели,
# чтобы понять значимость каждого параметра
drawFeatures(model, columns = X_train.columns)

# подготавливаем тестовые данные
poolResult = Pool(test, cat_features = categoricalFeatures)
# получаем предсказанные значения
result = model.predict(poolResult)

# записываем числовой результат предсказания в поле PATIENT_ID_COUNT
test['PATIENT_ID_COUNT'] = result.astype(int)

# сбрасываем все добавленные для обучения дополнительные поля
dropAugmentation(test, withClassify = False)

# сохраняем результат в файл, чтобы отправить на leaderboard
test.to_csv('result_pure.csv', sep=';', index=None)