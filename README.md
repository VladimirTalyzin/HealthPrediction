# HealthPrediction
Предсказание количества людей с различными заболеваниями. Для чемпионата по искусственному интеллекту.


**solution.py** - файл получения решения. При запуске создаёт result_pure.csv

**optimization.py** - поиск лучших комбинаций параметров CatBoots. У CatBoost есть свой поиск, но он работает не так, как нужно - не сравнивает по заданной функции с тестовой выборкой. Данное решение делает это

**prepare_data.py** - замена в решении тех наборов параметров, для которых было слишком мало данных на последние значения из тренировочной выборки. При достижении моделью CatBoost значения >0.9, это даёт примерно +0.002 (две тысячные) к Score. Важно скорее с исследовательской точки зрения, чтобы сделать вывод, что не стоит городить немыслимые модели там, где данных мало. Был ли вывод верным, можно будет понять, когда будет открыт private board. Если меня откинет ниже более, чем в среднем среди топ 15, значит предположение неверно. Иначе - верно.
Создаёт файлы result_prepared_last.csv (значения из последних значений), result_prepared_mean.csv (значения из средних значений), 
resultMean2022Empty (значения из средних значений за 2022 год, иначе - последних), result_prepared_mean_2022_empty.csv (значения из средних значений за 2022 год, иначе - последних, если встречается набор параметров, отсутствующий в файле обучения, то береётся среднее за 2022 год по упрощённому ключу из обучения). 

Наибольшее значение (+0.002) давал файл result_prepared_last.csv, он и был отправлен на leaderboard.

Утилиты:

**augmentation.py** - добавление дополнительных данных. Тип города, численность населения города, классификатор болезней, наличие данных за 2022 год. Классификатор болезней и 2022 год были отброшены в процессе исследования, как не дающие улушчений модели. Из этого можно сделать вывод, что классификатор болезней не отражает истинную взаимосвязь разных заболеваний

**utility.py** - всякие неважные подпрограммы


![Важность параметров](https://github.com/VladimirTalyzin/HealthPrediction/blob/main/features.png?raw=true)

Любопытна полученная высокая значимости типа города - крупный/мелкий/деревня/ещё мельче. Ранее я думал, может стоит переехать в маленький город. Теперь понимаю, что это не самая лучшая идея.
