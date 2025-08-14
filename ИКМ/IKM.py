import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1 (1)
data = pd.read_csv('Movies_data.csv', delimiter=',')
print(data.head())

#2 (1)
print("\nПропуски в данных:")
print(data.isnull().sum())

data['Certificate'].fillna('Unknown', inplace=True)
data['Subgenre'].fillna('Not Specified', inplace=True)
data['Subgenre 1'].fillna('Not Specified', inplace=True)
data['Meta_score'].fillna(data['Meta_score'].mean(), inplace=True)
data['Gross'].fillna(data['Gross'].mean(), inplace=True)

#3 (1)
print("\nТипы данных:")
print(data.dtypes)

data['Released_Year'] = pd.to_numeric(data['Released_Year'], errors='coerce')

numerical = ['IMDB_Rating', 'Meta_score', 'Gross', 'Runtime']
categorical = ['Certificate', 'Genre', 'Subgenre']

#4.1 (1)
print("\nФильмы жанра Crime:")
print(data[data['Genre'] == 'Crime'])

#4.2
print("\nФильмы с рейтингом IMDB выше или равным 9:")
print(data[data['IMDB_Rating'] >= 9])

# 5.1 (1)
print("\nСредний рейтинг IMDB:", data['IMDB_Rating'].mean())

#5.2
print("Общий доход всех фильмов:", data['Gross'].sum())

#6.1 (1)
print("\nФильм с максимальным рейтингом IMDB:")
print(data[data['IMDB_Rating'] == data['IMDB_Rating'].max()])

#6.2
print("\n5 фильмов с минимальным количеством голосов:")
print(data.nsmallest(5, 'No_of_Votes'))

#7 (1)
data['Certificate'].value_counts().plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title('Количество фильмов по сертификату')
plt.xlabel('Сертификат')
plt.ylabel('Количество')
plt.show()

#8 (1)
data['Genre'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Распределение жанров')
plt.show()

#9 (2) 
grouped_data = data.groupby('Genre').agg({'IMDB_Rating': ['mean', 'max'], 'Gross': ['mean', 'max']})
print("\nГруппировка по жанрам:")
print(grouped_data)

#10 (1)
data['High_Rating'] = (data['IMDB_Rating'] > 8.5).astype(int)

#11 (2)
sns.pairplot(data, vars=['IMDB_Rating', 'Meta_score', 'Gross'])
plt.show()

#12 (3)
for priznak in numerical:
    plt.figure(figsize=(10, 6))  
    sns.boxplot(data[priznak])
    plt.show()

Q1 = data['Gross'].quantile(0.25)
Q3 = data['Gross'].quantile(0.75)
IQR = Q3 - Q1
up = Q3 + 1.5 * IQR
data['Gross'] = data['Gross'].apply(lambda x: up if x > up else x)

#13
target = 'High_Rating'
features = ['Runtime', 'Meta_score', 'Gross']

# 14 (1)
data = pd.get_dummies(data, columns=['Genre'], drop_first=True)

# 15 (1)
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 16 (1)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)