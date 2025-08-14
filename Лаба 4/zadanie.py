import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

data = pd.read_csv('weather1.csv', usecols=['Местное время', 'T', 'P', 'U', 'Ff', 'N', 'H', 'VV'])

# Просмотр первых нескольких строк данных
print(data.head())


# Точечная диаграмма по признакам температуры и относительной влажности
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='T', y='U')
plt.title('Диаграмма рассеяния: Температура vs Относительная влажность')
plt.xlabel('Температура (°C)')
plt.ylabel('Относительная влажность (%)')
plt.grid()
plt.show()
# Условие для выделения точек
data['Color'] = data['N'].apply(lambda x: 'blue' if x == 100 else 'red')

# Построение точечной диаграммы с цветами
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='T', y='U', hue='Color')
plt.title('Температура vs Относительная влажность с выделением облачности')
plt.xlabel('Температура (°C)')
plt.ylabel('Относительная влажность (%)')
plt.grid()
plt.legend(title='Облачность')
plt.show()
# Преобразование столбца времени в datetime
data['Местное время'] = pd.to_datetime(data['Местное время'])

# Построение линейной диаграммы
plt.figure(figsize=(14, 7))
plt.plot(data['Местное время'], data['T'], marker='o')
plt.title('Изменение температуры по местному времени')
plt.xlabel('Местное время')
plt.ylabel('Температура (°C)')
plt.xticks(rotation=45)
plt.grid()
plt.show()
# Создание столбца с номером месяца
data['Месяц'] = data['Местное время'].dt.month

# Среднемесячная температура
monthly_avg_temp = data.groupby('Месяц')['T'].mean().reset_index()

# Построение столбчатой диаграммы
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_avg_temp, x='Месяц', y='T', palette='viridis')
plt.title('Средняя температура по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Средняя температура (°C)')
plt.xticks(ticks=range(12), labels=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
plt.grid()
plt.show()
# Построение ленточной диаграммы
cloudiness_counts = data['N'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(y=cloudiness_counts.index.astype(str), x=cloudiness_counts.values, orient='h')
plt.title('Количество наблюдений для каждого варианта облачности')
plt.xlabel('Количество наблюдений')
plt.ylabel('Облачность')
plt.grid()
plt.show()
# Построение гистограммы
plt.figure(figsize=(10, 6))
sns.histplot(data['T'], bins=10, kde=True)
plt.title('Гистограмма частот для температуры')
plt.xlabel('Температура (°C)')
plt.ylabel('Частота')
plt.grid()
plt.show()
# Разделение данных на группы по горизонтальной дальности видимости
data['Visibility Group'] = pd.cut(data['VV'], bins=[-float('inf'), 5, 15, float('inf')], labels=['<5 km', '5-15 km', '>15 km'])

# Построение boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Visibility Group', y='P')
plt.title('Атмосферное давление в зависимости от группы видимости')
plt.xlabel('Группа видимости')
plt.ylabel('Атмосферное давление (мм.рт.ст.)')
plt.grid()
plt.show()
# Подсчет уникальных значений высоты основания облаков
cloud_height_counts = data['H'].value_counts()

# Построение круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(cloud_height_counts, labels=cloud_height_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
