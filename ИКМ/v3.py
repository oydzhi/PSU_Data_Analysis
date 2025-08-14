import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Загрузка данных в датафрейм
df = pd.read_csv('v3.csv')

# Вывод нескольких строк данных, чтобы убедиться, что они загружены корректно
print("\nВывод нескольких строк данных:")
print(df.head())

# 2. Проверка пропущенных значений
print("\nКоличество пропущенных значений по каждому признаку:")
print(df.isnull().sum())

# Обработка пропущенных данных:
#  заменить пропуски для числовых столбцов медианой, для категориальных — значением 'UNKNOWN'
df['weight(kg)'] = df['weight(kg)'].fillna(df['weight(kg)'].median()) # пропущенные значения в столбце 'weight(kg)' заменяются медианным значением этого столбца
df['height(cm)'] = df['height(cm)'].fillna(df['height(cm)'].median())
df['field_position'] = df['field_position'].fillna('Unknown') # Все пропущенные значения в столбце 'field_position' заменяются строкой 'Unknown'.
df['nationality'] = df['nationality'].fillna('Unknown')
df.update(df['position'].fillna('UNKNOWN'))

print("\nОбработка пропущенных данных:")
print(df.head())

# 3. Определение типа признаков
print("\nОпределение типа признаков:")
print(df.dtypes)

# Преобразование типов, если необходимо
df['id_player'] = df['id_player'].astype(str) # преобразуем данные в столбце в строковый тип (string)
df['age'] = df['age'].astype(int) # преобразуем данные в столбце в целочисленный тип (integer)

# Определение числовых и категориальных признаков
numerical_features = ['weight(kg)', 'height(cm)', 'age'] # числовые признаки
categorical_features = ['nationality', 'field_position', 'position'] # категориальные признаки

# 4. Фильтрация данных: вопросы для фильтрации
# Вопрос 1: Вывести всех игроков, у которых позиция GOALKEEPER
goalkeeper_position = df[df['position'] == 'GOALKEEPER']
print("\nВсе игроки, у которых позиция GOALKEEPER:")
print(goalkeeper_position)

# Вопрос 2: Вывести всех игроков, возраст которых меньше 18 лет
print("\nВсе игроки, возраст которых меньше 18 лет:")
young_players = df[df['age'] < 18]
print(young_players)

# 5. Агрегирующие функции
# Вопрос 1: Средний возраст игроков
average_age = df['age'].mean()
print(f"\nСредний возраст игроков: {average_age}")

# Вопрос 2: Максимальный вес среди игроков
max_weight = df['weight(kg)'].max() 
print(f"\nМаксимальный вес игрока: {max_weight}")

# 6. Поиск минимальных и максимальных значений
# Вопрос 1: Игроки с максимальным ростом
max_height_player = df[df['height(cm)'] == df['height(cm)'].max()] # проверяем равняется ли значение в столбце 'height(cm)' максимальному значению в этом столбце
print("\nИгроки с максимальным ростом:")
print(max_height_player)

# Вопрос 2: Игроки с минимальным возрастом
min_age_player = df[df['age'] == df['age'].min()]
print("\nИгроки с минимальным возрастом:")
print(min_age_player)

# 7. Ленточная диаграмма для категориального признака по field_position
field_position_counts = df['field_position'].value_counts() # возвращаем количество уникальных значений в столбце и их частоту
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='field_position', order=field_position_counts.index) # sns.countplot Функция для создания ленточной диаграммы
plt.title('Распределение игроков по позициям')
plt.xticks(rotation=20)
plt.show()

# 8. Круговая диаграмма для другого категориального признака nationality
nationality_counts = df['nationality'].value_counts()
plt.figure(figsize=(8, 8))
nationality_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3') # plot.pie mетод pandas для построения круговой диаграммы
plt.title('Распределение игроков по национальностям')
plt.ylabel('')
plt.show()

# 9. Группировка данных по категориальному признаку field_position
position_stats = df.groupby('field_position')[['weight(kg)', 'height(cm)']].agg(['mean', 'max']) # группируем данные по значению столбца 'field_position' и выполненяем агрегирующие функции
print("\nГруппировка данных по категориальному признаку field_position:")
print(position_stats)

# 10. Добавление нового признака (бинарный признак "высокий игрок" по росту)
df['tall_player'] = np.where(df['height(cm)'] > 185, 1, 0)
# Проверка добавленного признака:
# Вывести несколько строк датафрейма
print("\nДобавление нового признака (бинарный признак высокий игрок по росту)")
print(df[['id_player', 'height(cm)', 'tall_player']].head())

# 11. Точечная диаграмма для числовых признаков
sns.pairplot(df[['weight(kg)', 'height(cm)', 'age']]) # строим матрицу диаграмм рассеяния для каждой пары выбранных числовых признаков
plt.show()

# 12. Boxplot для числовых признаков
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['weight(kg)', 'height(cm)', 'age']])
plt.title('Boxplot для числовых признаков')
plt.show()

# Определение выбросов для выбранного признака вес
print("\nСтатистика до замены выбросов:")
print(df['weight(kg)'].describe()) # Функция для получения статистического описания столбца

# До замены выбросов
plt.figure(figsize=(10, 6))
sns.boxplot(df['weight(kg)'])
plt.title('Boxplot до замены выбросов')
plt.show()

Q1 = df['weight(kg)'].quantile(0.25)
Q3 = df['weight(kg)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nНижняя граница выбросов: {lower_bound}")
print(f"\nВерхняя граница выбросов: {upper_bound}")

# Заменяем выбросы на максимальное значение, которое не является выбросом
df['weight(kg)'] = np.where(df['weight(kg)'] < lower_bound, upper_bound, df['weight(kg)']) # Условие для проверки, является ли значение выбросом (меньше нижней границы)
df['weight(kg)'] = np.where(df['weight(kg)'] > upper_bound, upper_bound, df['weight(kg)']) # Условие для проверки, является ли значение выбросом (больше верхней границы)

# После замены выбросов
print("\nСтатистика после замены выбросов:")
print(df['weight(kg)'].describe())

# После замены выбросов
plt.figure(figsize=(10, 6))
sns.boxplot(df['weight(kg)'])
plt.title('Boxplot после замены выбросов')
plt.show()

# 13. Задача для кластеризации
# Кластеризация игроков по возрасту и росту
X = df[['age', 'height(cm)']].dropna()  # Преобразуем данные и убираем пропуски
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Вывод данных с метками кластеров
print("\nДанные с метками кластеров:")
print(df[['age', 'height(cm)', 'cluster']].head())

# 14. Список факторных признаков
factor_columns = ['nationality', 'field_position', 'position']

# Проверка факторных признаков
print("\nСписок факторных признаков и их уникальные значения:")
for col in factor_columns:
    print(f"\n{col}: {df[col].unique()}")

# 15. Кодирование категориальных признаков
df_encoded = pd.get_dummies(df, columns=factor_columns, drop_first=True) # кодирования категориальных признаков с использованием one-hot encoding
# Вывод результатов
print("\nИсходные данные:")
print(df)
print("\nДанные после кодирования:")
print(df_encoded)

# 16. Нормализация числовых признаков
scaler = StandardScaler() # создает объект для нормализации данных
df[['weight(kg)', 'height(cm)', 'age']] = scaler.fit_transform(df[['weight(kg)', 'height(cm)', 'age']]) # сначала обучает (вычисляет средние значения и стандартные отклонения) на основе данных, а затем применяет трансформацию к данным, нормализуя их.
print("\nДанные после нормализации:")
print(df)
# Проверка средней и стандартного отклонения для каждого признака
print("\nСтатистика после нормализации:")
for column in ['weight(kg)', 'height(cm)', 'age']:
    print(f"{column}: Среднее = {df[column].mean():.2f}, Стандартное отклонение = {df[column].std():.2f}")

