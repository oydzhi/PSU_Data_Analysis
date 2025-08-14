import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('athlete_events (2).csv')

value_num= data.count()
print(value_num)
data.info()

missing_data = data.isnull().sum()
print("Признаки с отсутствующими данными:\n", missing_data[missing_data > 0])
most_missing = missing_data.idxmax(), missing_data.max()
print("Признак с наибольшим количеством отсутствующих данных:", most_missing)

stat = data[['Age', 'Height', 'Weight']].describe()
print(stat)

young1992 = data[data['Year'] == 1992].nsmallest(1, 'Age')
print(young1992[['Name', 'Age', 'Event']])

print("Все виды спорта:", data['Sport'].nunique())

print("Средний рост теннисисток, учавствовавших в 2000 году", 
data[(data['Sex'] == 'F') & (data['Sport'] == "Tennis") & (data['Year'] == 2000)]['Height'].mean())

print("Количество золотых медалей в настольном теннисе для Китая в 2008 году:", 
data[(data['NOC'] == 'CHN') & (data['Year'] == 2008) & (data['Sport'] == 'Table Tennis') & (data['Medal'] == 'Gold')].shape[0])

sport2004 = data[(data['Year'] == 2004) & (data['Season'] == 'Summer')]['Sport'].nunique()
sport1988 = data[(data['Year'] == 1988) & (data['Season'] == 'Summer')]['Sport'].nunique()
diff = sport2004 - sport1988
print("Изменение количества видов спорта между 1988 и 2004:", diff)

curling2014 = data[(data['Sport'] == 'Curling') & (data['Year'] == 2014) & (data['Sex'] == 'M')]
plt.hist(curling2014['Age'], bins=10, edgecolor='black')
plt.title('Распределение возраста мужчин-керлингистов (2014)')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

medals_age_2006 = data[data['Year'] == 2006].groupby('NOC').agg({'Medal': 'count', 'Age': 'mean'})
print(medals_age_2006[medals_age_2006['Medal'] > 0])

medals_pivot = data[data['Year'] == 2006].pivot_table(index='NOC', columns='Medal', aggfunc='size', fill_value=0)
print(medals_pivot)
