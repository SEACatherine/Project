
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Витрина данных",
        page_icon="👋",
    )

    st.write("Добро пожаловать 👋")

    st.sidebar.success("Тест sidebar")

    st.markdown(
    """
    Добро пожаловать в наше приложение для анализа акустических данных! 👋

    Это приложение предназначено для анализа данных о голосе и акустических сигналах. Вы можете использовать его для изучения и визуализации различных аспектов данных о голосе.

    **👈 Выберите демонстрацию в боковой панели**, чтобы увидеть примеры анализа данных и результаты моделей машинного обучения.

    ### Главные функции приложения:

    **DataFrame**
    - Эта страница отображает таблицу данных используемого датасета.

    **Heatmap: Тепловая карта корреляций**
    - Эта страница отображает тепловую карту корреляций между разными акустическими признаками в данных.

    **Median graph: График медианы диапазона доминирующей частоты**
    - Эта страница включает график медианы для диапазона доминирующей частоты в акустическом сигнале.

    **Median and maximum value graph: Сравнение медианы и максимального значения диапазона доминирующей частоты**
    - Здесь вы найдете график, сравнивающий медиану и максимальное значение для диапазона доминирующей частоты.

    **Расстояние от STD и MEAN для *Mean Dom*: Расстояние от STD и MEAN для Mean Dom**
    - Эта страница визуализирует расстояние от стандартного отклонения (STD) и среднего значения (MEAN) для параметра "Mean Dom".

    **Расстояние от STD и MEAN для *IQR*: Расстояние от STD (стандартное отклонение) и MEAN (среднее значение) для IQR**
    - Здесь вы найдете график, отображающий расстояние от STD и MEAN для параметра "IQR".

    **Распределение данных и их связь**
    - На этой странице показано распределение данных и их связь между "q25" и "q75", а также их связь с категорией "label (male, female)".

    **Анализ корреляции с "label"**
    - Здесь представлены столбцы с наибольшей корреляцией с колонкой "label".

    **Средняя частота**
    - Эта страница отображает распределение средней частоты для "meanfun".

    **Диапазон доминирующей частоты**
    - Здесь вы найдете график, представляющий распределение диапазона доминирующей частоты в акустическом сигнале.

    **KDE: KDE-графики для каждого признака**
    - На этой странице отображаются графики оценки плотности ядра (KDE) для каждого признака, разделенные по классам (female = 0, male = 1).

    **Обучение, тестирование и результаты моделей**
    - На странице "Модель логистической регрессии" представлены результаты обучения модели логистической регрессии, включая максимальную и минимальную точность.
    - На странице "Нейронная модель" вы увидите результаты обучения нейронной сети, включая точность и оценку производительности на тестовых данных, а также результаты бинарных предсказаний.
    - Страница "Визуализация производительности двух моделей" отображает графики производительности двух моделей с использованием кривых ROC и площади под ними (AUC).
    - На странице "Графики плотности вероятности для двух моделей" вы найдете графики плотности вероятности для двух моделей.
    - На странице "Model Testing" вы можете исследовать и оценивать различные модели машинного обучения на своих собственных данных.
    """
)

if __name__ == "__main__":
    run()
