from csv import DictReader, writer
from augmentation import is2022, dateToMonths

# Файл обучающей выборки
train  = "train.csv"
# Файл предсказанных данных, полученный моделью, который берётся за основу постобработки
baseFile = "result_pure.csv"
# Файл в котором значения, где было мало данных, заменены на последнее известное значение
resultLast = "result_prepared_last.csv"
# Файл в котором значения, где было мало данных, заменены на среднее известное значение
resultMean = "result_prepared_mean.csv"
# Файл в котором значения, где было мало данных, заменены на среднее известное значение за 2022 год. Если за 2022 год не было значений, тогда на последнее известное
resultMean2022 = "result_prepared_mean_2022.csv"
# Значения из средних значений за 2022 год, иначе - последних,
# если встречается набор параметров, отсутствующий в файле обучения, то берётся среднее за 2022 год по упрощённому ключу из обучения
resultMean2022Empty = "result_prepared_mean_2022_empty.csv"

# Уровень количества данных. Если данных меньше заданного уровня, то происходит преобразование
NO_DATA_LEVEL = 3

# Собранный список обучающих данных
trainData = []

# Обучающие данные, собранные по ключу (все данные, за исключением месяца)
keyFields = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY']
collectData = {}

# Обучающие данные, собранные по упрощённому ключу (все данные, за исключением месяца и пола пациента)
# то, что пол является самым незначимым параметром данных, было установлено из графиков значимости параметров
# вероятно, болезни в значительной степени уже распределены по полу
lowDataFields = ['MKB_CODE', 'ADRES', 'AGE_CATEGORY']
collectLowData = {}

# подготавливаем данные
# считаем количество данных для каждого ключа,
# список всех значений и список всех значений за 2022 год,
# а также значение, самое последнее по времени
def prepareData(currentValue, monthNumber, data, rowIndex):
    if rowIndex in data:
        data[rowIndex]["count"] += 1
        data[rowIndex]["allValues"].append(currentValue)

        if is2022(monthNumber):
            data[rowIndex]["all2022Values"].append(currentValue)

        if monthNumber < data[rowIndex]["minDate"]:
            data[rowIndex]["minDate"] = monthNumber
            data[rowIndex]["lastValue"] = currentValue

    else:
        data[rowIndex] = {"count": 1, "lastValue": currentValue, "minDate": monthNumber,
                                 "allValues": [currentValue],
                                 "all2022Values": [currentValue] if is2022(monthNumber) else []}

# устанавливаем поля со средним значением
# и со средним значением за 2022 год
def prepareMean(data):
    for rowIndex, values in data.items():
        values["mean"]     = max(1, int(round(sum(values["allValues"]) / len(values["allValues"]))))
        if len2022 := len(values["all2022Values"]) > 0:
            values["mean2022"] = max(1, int(round(sum(values["all2022Values"]) / len2022)))
        else:
            values["mean2022"] = values["lastValue"]


# считываем файл
with open(train, "r", encoding = "utf8") as csvFile:
    reader = DictReader(csvFile, delimiter=';')

    for row in reader:
        # сохраняем все прочитанные значения
        trainData.append(row)
        # вычисляем ключи данных
        rowIndex = "|".join(map(str, [row[key] for key in keyFields]))
        lowDataIndex = "|".join(map(str, [row[key] for key in lowDataFields]))

        # определяем количество заболевших
        currentValue = int(row["PATIENT_ID_COUNT"])
        # определяем количество месяцев с 04.2022
        monthNumber = dateToMonths(row["VISIT_MONTH_YEAR"])

        # сохраняем данные по соответствующим ключам
        prepareData(currentValue, monthNumber, collectData, rowIndex)
        prepareData(currentValue, monthNumber, collectLowData, lowDataIndex)

# создаём поля со средними значениями
prepareMean(collectData)
prepareMean(collectLowData)

# получаем список ключей, в которых данных меньше, чем заданное количество
itemData = []
for index in range(NO_DATA_LEVEL):
    itemData.append(dict((key, value) for key, value in collectData.items() if value["count"] == index + 1))

# выводим информацию о том, сколько всего значений, сколько всего ключей значений
# и сколько ключей значений содержит данных меньше заданного уровня
print("All train:", len(trainData), "all collect:", len(collectData))
for index in range(NO_DATA_LEVEL):
    print(index + 1, "item:", len(itemData[index]))


# начинаем собирать итоговые данные
resultData = {}
replaceLast = [{} for _ in range(NO_DATA_LEVEL)]
replaceMean = [{} for _ in range(NO_DATA_LEVEL)]
replaceMean2022 = [{} for _ in range(NO_DATA_LEVEL)]
replaceEmpty = {}

# считываем файл с предсказаниями, полученными от модели
with open(baseFile, "r", encoding = "utf8") as csvFile:
    reader = DictReader(csvFile, delimiter=';')

    idRow = 0
    for row in reader:
        # сохраняем значение строки
        resultData[idRow] = row
        # вычисляем ключи данных этой строки
        rowIndex = "|".join(map(str, [row[key] for key in keyFields]))
        lowDataIndex = "|".join(map(str, [row[key] for key in lowDataFields]))

        # определяем предсказанное количество пациентов
        currentValue = int(row["PATIENT_ID_COUNT"])

        for index in range(NO_DATA_LEVEL):
            # если ключ в списке ключей, с недостаточным количество данных:
            if rowIndex in itemData[index]:
                # берём последнее значение
                lastValue = itemData[index][rowIndex]["lastValue"]

                # если предсказанное значение отличается, то сохраняем нужное значение в соответствующий список
                if lastValue != currentValue:
                    replaceLast[index][idRow] = lastValue
                    break

                # берём среднее значение
                meanValue = itemData[index][rowIndex]["mean"]
                # если предсказанное значение отличается, то сохраняем нужное значение в соответствующий список
                if meanValue != currentValue:
                    replaceMean[index][idRow] = meanValue
                    break

                # берём среднее за 2022 год значение
                mean2022Value = itemData[index][rowIndex]["mean2022"]
                # если предсказанное значение отличается, то сохраняем нужное значение в соответствующий список
                if mean2022Value != currentValue:
                    replaceMean2022[index][idRow] = mean2022Value
                    break

            # если такого ключа вообще не было в тренировочных данных
            elif not rowIndex in collectData and lowDataIndex in collectLowData:
                # то сохраняем для него среднее значение за 2022 год
                replaceEmpty[idRow] = collectLowData[lowDataIndex]["mean2022"]

        idRow += 1


# выводим количество всех значений и всех значений, где будет заменено количество пациентов
print("All result:", len(resultData))

for index in range(NO_DATA_LEVEL):
    print("replace last", index + 1, ":", len(replaceLast[index]))

for index in range(NO_DATA_LEVEL):
    print("replace mean", index + 1, ":", len(replaceMean[index]))

for index in range(NO_DATA_LEVEL):
    print("replace mean 2022", index + 1, ":", len(replaceMean2022[index]))

print("replace empty:", len(replaceEmpty))


# сохранить обработанные данные в файл
def saveNewResult(resultCopyData, newFilename):
    resultColumns = ["PATIENT_SEX", "MKB_CODE", "ADRES", "VISIT_MONTH_YEAR", "AGE_CATEGORY", "PATIENT_ID_COUNT"]
    with open(newFilename, "w", encoding = "utf8", newline = "") as csvFile:
        csvWriter = writer(csvFile, delimiter = ";")
        csvWriter.writerow(resultColumns)
        for _, row in resultCopyData.items():
            csvWriter.writerow([row[key] for key in resultColumns])


# обработать и сохранить данные
def saveResult(filename, replaceData, empty):
    # создаём копию данных, полученных от модели
    resultCopyData = resultData.copy()

    # для всех подготовленных замен выполняем замену количества пациентов
    for index in range(NO_DATA_LEVEL):
        for idRow, value in replaceData[index].items():
            resultCopyData[idRow]["PATIENT_ID_COUNT"] = value

    # если указано, заменять все те данные, ключей которых не было в обучающей выборке, то заменяем и их
    if empty is not None:
        for idRow, value in empty.items():
            resultCopyData[idRow]["PATIENT_ID_COUNT"] = value

    # заменяем все значения, <= 0 на 1. Исходя из того, что тестовая выборка является исследованием, а следовательно,
    # там только те, кто обследовался
    zerosCounter = 0
    for idRow, row in resultCopyData.items():
        if int(row["PATIENT_ID_COUNT"]) <= 0:
            row["PATIENT_ID_COUNT"] = 1
            zerosCounter += 1

    # выводим, сколько значений <= 0 было заменено на 1
    print("replace zeros:", zerosCounter)

    # сохраняем результат в файл
    saveNewResult(resultCopyData, filename)


# заменяем и сохраняем по последнему значению (* лучшее решение, отправляемое в итоге на leaderboard)
saveResult(resultLast, replaceLast, None)
# заменяем и сохраняем по среднему значению
saveResult(resultMean, replaceMean, None)
# заменяем и сохраняем по среднему за 2022 год значению
saveResult(resultMean2022, replaceMean2022, None)
# заменяем и сохраняем по среднему за 2022 год значению,
# заменяя также и все данные, с ключами, отсутствующими в тренировочной выборке
saveResult(resultMean2022Empty, replaceMean2022, replaceEmpty)