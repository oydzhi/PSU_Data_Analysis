import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols_to_read = ['Местное время в Перми', 'T', 'P', 'U', 'Ff', 'N', 'H', 'VV']
data = pd.read_csv('Laba5.csv', sep=';', usecols=cols_to_read)

#print(data.head())
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='T', y='U')
plt.title('Диаграмма рассеяния: Температура vs Относительная влажность')
plt.xlabel('Температура (°C)')
plt.ylabel('Относительная влажность (%)')
plt.grid()
plt.show()



data['Color'] = data['N'].apply(lambda x: 'blue' if x == 100 else 'red')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='T', y='U', hue='Color')
plt.title('Температура vs Относительная влажность с выделением облачности')
plt.xlabel('Температура (°C)')
plt.ylabel('Относительная влажность (%)')
plt.grid()
plt.legend(title='Облачность')
plt.show()



data['Местное время в Перми'] = pd.to_datetime(data['Местное время в Перми'])
plt.plot(data['Местное время в Перми'], data['T'], marker='o')
plt.title('Изменение температуры по местному времени')
plt.xlabel('Местное время')
plt.ylabel('Температура (°C)')
plt.xticks(rotation=45)
plt.grid()
plt.show()



data['Месяц'] = data['Местное время в Перми'].dt.month
monthly_avg_temp = data.groupby('Месяц')['T'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_avg_temp, x='Месяц', y='T', palette='viridis')
plt.title('Средняя температура по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Средняя температура (°C)')
plt.xticks(ticks=range(12), labels=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
plt.grid()
plt.show()



cloudiness_counts = data['N'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(y=cloudiness_counts.index.astype(str), x=cloudiness_counts.values, orient='h')
plt.title('Количество наблюдений для каждого варианта облачности')
plt.xlabel('Количество наблюдений')
plt.ylabel('Облачность')
plt.grid()
plt.show()



plt.figure(figsize=(10, 6))
sns.histplot(data['T'], bins=10, kde=True)
plt.title('Гистограмма частот для температуры')
plt.xlabel('Температура (°C)')
plt.ylabel('Частота')
plt.grid()
plt.show()



data['Visibility Group'] = pd.cut(data['VV'], bins=[-float('inf'), 5, 15, float('inf')], labels=['<5 km', '5-15 km', '>15 km'])
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Visibility Group', y='P')
plt.title('Атмосферное давление в зависимости от группы видимости')
plt.xlabel('Группа видимости')
plt.ylabel('Атмосферное давление (мм.рт.ст.)')
plt.grid()
plt.show()



cloud_height_counts = data['H'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(cloud_height_counts, labels=cloud_height_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
plt.title('Распределение высоты основания облаков')
plt.axis('equal')
plt.show()