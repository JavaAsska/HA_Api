import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
def find_binary(df, max_unique=2):
    """Определяет бинарные признаки в DataFrame на основе количества уникальных значений.

    Бинарными считаются признаки, у которых количество уникальных значений (исключая пропуски)
    не превышает заданного порога (по умолчанию 2).

    Args:
        df (pd.DataFrame): DataFrame для анализа.
        max_unique (int, optional): Максимальное количество уникальных значений для признака,
                                  чтобы считаться бинарным. По умолчанию 2.

    Returns:
        list: Список названий столбцов, которые являются бинарными признаками.
    """
    binary_features = []
    for column in df.columns:
        unique_values = df[column].dropna().nunique()
        if unique_values <= max_unique:
            binary_features.append(column)
    return binary_features
sns.set_style('whitegrid')
sns.set_palette('RdYlBu')
def graph_num(df, columns, title, xlabel, bins=30):
    """
    Визуализирует числовые данные с помощью гистограммы и boxplot.

    Параметры:
    -----------
    df : pandas.DataFrame
        DataFrame, содержащий данные для анализа.
    columns : str
        Название столбца с числовыми данными, которые будут визуализированы.
    title : str
        Заголовок графика (добавляется к описанию гистограммы и boxplot).
    xlabel : str
        Подпись оси X для гистограммы.
    bins : int, optional
        Количество интервалов (бинов) для гистограммы. По умолчанию 30.

    
    """
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    sns.histplot(data=df, x=columns, kde=True, color='blue', bins=bins, ax=ax[0])
    ax[0].set_title('Гистограмма признака '+ title)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('Кол-во')
    
    sns.boxplot(data=df[columns], orient='h', ax=ax[1])
    ax[1].set_title('Разброс значений признака '+ title)
    ax[1].set_xlabel('Значения')
    ax[1].set_ylabel(' ')
def check_data_dubmiss(df, df_name='DataFrame'):
    """
    Анализирует DataFrame на дубликаты и пропущенные значения.
    Возвращает единый отчет в виде DataFrame с категориями.

    Параметры:
    - df: pandas DataFrame для проверки
    - df_name: название DataFrame для отчета

    Возвращает:
    - Единый DataFrame с отчетом
    """
    report_parts = []

    # 1. Общая информация
    info_df = pd.DataFrame({
        'Категория': ['Общая информация'],
        'Параметр': ['Название датафрейма'],
        'Значение': [df_name],
        'Детали': [f'Размер : {df.shape[0]} x {df.shape[1]}']
    })
    report_parts.append(info_df)

    # 2. Дубликаты
    duplicates = df.duplicated().sum()
    duplicates_df = pd.DataFrame({
        'Категория': ['Дубликаты', 'Дубликаты'],
        'Параметр': ['Всего дубликатов', 'Примеры дубликатов'],
        'Значение': [duplicates, duplicates > 0],
        'Детали': ['', f"Первые {min(5, duplicates)} примеров" if duplicates > 0 else '']
    })
    report_parts.append(duplicates_df)

    # 3. Пропущенные значения по столбцам
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Категория': ['Пропущенные значения'] * len(missing),
            'Параметр': missing.index,
            'Значение': missing.values,
            'Детали': [f"{pct:.2f}%" for pct in missing_pct.values]
        })
        missing_df = missing_df[missing_df['Значение'] > 0]
        report_parts.append(missing_df)

    # 4. Итог по пропущенным значениям
    total_missing_df = pd.DataFrame({
        'Категория': ['Пропуски'],
        'Параметр': ['Всего пропущенных значений'],
        'Значение': [missing.sum()],
        'Детали': [f"{missing.sum() / df.size:.2%} от всех ячеек"]
    })
    report_parts.append(total_missing_df)

    report = pd.concat(report_parts, ignore_index=True)

    if duplicates > 0:
        duplicates_examples = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head()
        report.at[4, 'Детали'] = str(duplicates_examples.to_dict('records'))

    return report
def one_screen_hist(data, columns, titles=None, xlabels=None, 
                        bins=30, figsize=None, color='blue', 
                        kde=True, title_prefix='Гистограмма признака '):
    """
    Строит сетку гистограмм для произвольного количества столбцов
    
    Параметры:
    ----------
    data : DataFrame
        Исходный датафрейм с данными
    columns : list of str
        Список названий столбцов для построения графиков
    titles : list of str, optional
        Список заголовков для графиков (по умолчанию используются имена столбцов)
    xlabels : list of str, optional
        Список подписей осей X (по умолчанию используются имена столбцов)
    bins : int, optional
        Количество бинов для гистограмм (по умолчанию 30)
    figsize : tuple, optional
        Размер фигуры (по умолчанию вычисляется автоматически)
    color : str, optional
        Цвет гистограмм (по умолчанию 'blue')
    kde : bool, optional
        Отображать ли KDE (по умолчанию True)
    title_prefix : str, optional
        Префикс для заголовков графиков (по умолчанию 'Гистограмма признака ')
    """
    n = len(columns)
    if n == 0:
        raise ValueError("Список columns не может быть пустым")
    
    # Установка значений по умолчанию
    if titles is None:
        titles = columns
    if xlabels is None:
        xlabels = columns
    
    # Проверка входных данных
    if len(titles) != n or len(xlabels) != n:
        raise ValueError("columns, titles и xlabels должны иметь одинаковую длину")
    
    # Вычисление оптимального размера сетки
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    
    # Автоматический подбор размера фигуры
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()  # Преобразуем в плоский массив
    
    for i, (col, title, xlabel) in enumerate(zip(columns, titles, xlabels)):
        sns.histplot(data=data, x=col, kde=kde, color=color, bins=bins, ax=axes[i])
        axes[i].set_title(f'{title_prefix}{title}')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Кол-во')
    
    # Скрываем пустые графики
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()