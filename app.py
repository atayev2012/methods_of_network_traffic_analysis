import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# Базовая конфигурация логирования сообщений уровня INFO и выше
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(asctime)s] - %(message)s')


class DataContext:
    """
    Класс DataContext представляет собой минимально требуемую информацию для обработки csv файла с данными
    """
    def __init__(
            self,
            csv_file_path: str | None = None,
            x_data_file_path: str = "x_data.npy",
            y_data_file_path: str = "y_data.npy",
            target_column: str = "",
    ):
        """
        Инициализирует новый объект DataContext

        Параметры:
        ----------
        csv_file_path :
            Путь к файлу с входными данными
        x_data_file_path :
            Путь к файлу данных признаков (по умолчанию x_data.npy)
        y_data_file_path :
            Путь к файлу данных целевой переменной (по умолчанию y_data.npy)
        target_column :
            Название колонки для целевой переменной (по умолчанию "")

        """
        # Пути к файлам
        self.csv_file_path = csv_file_path
        self.x_data_file_path = x_data_file_path
        self.y_data_file_path = y_data_file_path

        # Табличные данные
        self.target_column = target_column # Столбец целевой переменной
        self.data = None # pd.DataFrame с исходными данными
        self.X = None # pd.DataFrame с признаками
        self.y = None # pd.DataFrame с целевой переменной

class Handler:
    """
    Класс Handler представляет собой базовый обработчик данных
    """
    def __init__(self, next_handler = None):
        """
        Инициализирует новый объект Handler

        Параметры:
        ----------
            next_handler: Handler
                Следующий обработчик в пайплайне
        """
        self.next_handler = next_handler

    def set_next(self, handler):
        """
        Устанавливает следующий обработчик

        Параметры:
        ----------
            handler: Handler | None
                Следующий обработчик в пайплайне
        """
        self.next_handler = handler
        return handler # Возвращается следующий обработчик (Handler)

    def handle(self, context: DataContext):
        """
        Запускает обработчик

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """
        self.process(context)

        # Запуск следующего обработчика в пайплайне если он существует
        if self.next_handler:
            self.next_handler.handle(context)

    def process(self, context: DataContext):
        """
        Метод обработчик

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """
        pass


class LoadFilePathHandler(Handler):
    """
    Класс LoadFilePathHandler представляет собой базовый обработчик пути к файлу с данными
    """
    def process(self, context: DataContext):
        """
        Метод обработчик для считывания пути к файлу с данными из аргументов скрипта

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """

        # Проверка входных параметров
        # Обработка переданных аргументов (считывается только первый аргумент остальные игнорируются)
        if len(sys.argv) < 2:
            raise ValueError(
                f"Недостаточно аргументов!\n"
                f"Использование: python app.py [PATH TO CSV FILE]"
            )

        # Обработка существования файла
        if not os.path.isfile(sys.argv[1]):
            raise FileNotFoundError(f"Файл {sys.argv[1]} не найден!")

        # Обновляем путь к файлу с данными
        context.csv_file_path = sys.argv[1]

        logging.info(f"Путь к файлу с данными успешно добавлен: {context.csv_file_path}")


class LoadCSVHandler(Handler):
    """
    Класс LoadCSVHandler представляет собой базовый обработчик подгрузки данных из csv файла
    """
    def process(self, context: DataContext):
        """
        Метод обработчик для подгрузки данных из csv файла

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """

        # Заполняем поле data данными DataFrame считанными из csv файла
        context.data = pd.read_csv(context.csv_file_path)

        logging.info(f"Данные из файла {context.csv_file_path} успешно считаны")


class DataPreprocessingHandler(Handler):
    """
    Класс DataPreprocessingHandler представляет собой обработчик-преобразователь данных в более понятный формат
    """

    # Словарь для преобразования месяцев
    months_dict = {
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    # Курсы валют к рублю в разных обозначениях
    currency_rates = {
        "РУБ.": 1,
        "KZT": 0.151579,
        "EUR": 96.8345,
        "RUB": 1,
        "USD": 83.0,
        "ГРН.": 2.0077,
        "AZN": 48.8235,
        "БЕЛ. РУБ.": 27.3116,
        "KGS": 0.949114,
        "СУМ": 0.00055148,
        "BYN": 27.3116,
        "UAH": 2.0077,
        "SOM": 0.949114
    }

    # Функция для обработки строки пола и возраста
    def parse_gender_age(self, row: pd.Series) -> pd.Series:
        """
        Метод парсинга данных пола и возраста

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """

        # Разбиваем строку на части
        parts = row.split(' , ')

        # Пол (оставляем как строку "Мужчина"/"Женщина")
        gender = parts[0].strip()

        # Возраст
        age_part = parts[1].strip()
        age = int(age_part.split()[0])

        # Дата рождения (собираем в дату)
        if gender == "Male" or gender == "Female":
            birth_part = parts[2].strip()

        else:
            birth_part = parts[2].strip()
            birth_parts = (birth_part.lower().replace("родился", "").replace("родилась", "").strip().split())

            day = int(birth_parts[0])
            month = self.months_dict[birth_parts[1]]
            year = int(birth_parts[2])

        # Создаем объект даты
        birth_date = datetime(year, month, day)

        # Возвращаем пол, возраст и дату рождения
        return pd.Series([gender, age, birth_date])

    def process(self, context: DataContext):
        """
        Метод обработчик для преобразования данных в более понятный формат

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """

        # Удаляем все пустые данные
        context.data = context.data.dropna()

        # Применяем функцию и создаем новые столбцы
        context.data[['Пол', 'Возраст', 'Дата рождения']] = context.data['Пол, возраст'].apply(self.parse_gender_age)

        # Удаляем оригинальный столбец
        context.data = context.data.drop(columns=['Пол, возраст'])

        # Обработка заработной платы.
        # Считываем все виды валют, убираем пробелы, возводим в верхний регистр и удаляем \xa0 (специальный пробел)
        currencies = context.data["ЗП"].str.replace(r"\d+\s+", "", regex=True).str.upper().str.replace("\xa0", " ")

        # Считываем все зарплаты в числовом виде
        salary = context.data["ЗП"].str.replace(r"\D+", "", regex=True)

        # Создаем новую колонку, где зарплаты в только в рублях
        context.data["ЗП В РУБЛЯХ"] = salary * currencies.map(self.currency_rates)


class EncodeCategoricalHandler(Handler):
    """
    Класс EncodeCategoricalHandler представляет собой обработчик-преобразователь категориальных колонок
    """
    def process(self, context: DataContext):
        """
        Метод обработчик для преобразования категориальных колонок

        Параметры:
        ----------
            context:
                Данные по которым запускается обработчик
        """
        df = context.data
        cat_cols = df.select_dtypes(include=["object"]).columns
        context.data = pd.get_dummies(df, columns=cat_cols)


class SplitXYHandler(Handler):
    """
    Класс SplitXYHandler представляет собой обработчик-разделитель на
    """
    def process(self, context: DataContext):
        df = context.data
        target = context.target_column

        context.X = df.drop(columns=[target]).values
        context.y = df[target].values


class ScaleXHandler(Handler):
    def process(self, context: DataContext):
        scaler = StandardScaler()
        context.X = scaler.fit_transform(context.X)


class SaveNumpyHandler(Handler):
    def process(self, context: DataContext):
        np.save(context.x_data_file_path, context.X)
        np.save(context.y_data_file_path, context.y)


def main():
    # Инициализация данных
    context_data = DataContext()

    # Создание пайплайна. Создаем шаги.
    path = LoadFilePathHandler()
    loader = LoadCSVHandler()
    preprocessor = DataPreprocessingHandler()
    encoder = EncodeCategoricalHandler()
    splitter = SplitXYHandler()
    scaler = ScaleXHandler()
    saver = SaveNumpyHandler()

    # Устанавливаем последовательность
    scaler.set_next(saver)
    splitter.set_next(scaler)
    encoder.set_next(splitter)
    preprocessor.set_next(encoder)
    loader.set_next(preprocessor)
    path.set_next(loader)


    # Запуск пайплайна
    path.handle(context_data)




if __name__ == "__main__":
    # main()
    context_data = DataContext()
    path = LoadFilePathHandler()
    loader = LoadCSVHandler()
    path.handle(context_data)
    loader.handle(context_data)
    print(context_data.data['Пол, возраст'].unique().tolist())