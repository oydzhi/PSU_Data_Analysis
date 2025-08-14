import pandas as pd

data = pd.read_csv('telecom_churn.csv')



#data.info()
#data.describe()



missing_data = data.isnull().sum()
#print("Отсутствующие данные:", missing_data[missing_data > 0])




churn_counts = data['Churn'].value_counts()
active = churn_counts[False] / len(data) * 100
unactive = churn_counts[True] / len(data) * 100

#print("Количество активных клиентов:", churn_counts[False])
#print("Количество потерянных клиентов:", churn_counts[True])
#print("Процент активных клиентов:", active, "%")
#print("Процент потерянных клиентов:", unactive,"%")



data['Average call duration'] = (data['Total day minutes'] + data['Total eve minutes'] + data['Total night minutes'])/(
                                data['Total day calls'] + data['Total eve calls'] + data['Total night calls'])

top10 = data.sort_values(by='Average call duration', ascending=False).head(10)
#print(top10[['Average call duration']])




avg_dur = data.groupby('Churn')['Average call duration'].mean()
#print(avg_dur)
#print("Разница в средней продолжительности:", avg_dur[True] - avg_dur[False])



avg_sup = data.groupby('Churn')['Customer service calls'].mean()
#print(avg_sup)
#print("Разница в среднем количестве звонков в службу поддержки:", avg_sup[True] - avg_sup[False])



crosstable_ = pd.crosstab(data['Customer service calls'], data['Churn'])
#print(crosstable)

crosstable_proc = crosstable_.div(crosstable_.sum(axis=1), axis=0) * 100
#print(crosstable_proc)
print("Количество звонков в службу поддержки с процентом оттока выше 40%:\n", crosstable_proc[crosstable_proc[True] > 40])



crosstable_inter = pd.crosstab(data['International plan'], data['Churn'])
print(crosstable_inter)

crosstable_inter_proc = crosstable_inter.div(crosstable_inter.sum(axis=1), axis=0) * 100
print(crosstable_inter_proc)
print("Процент оттока среди клиентов с международным роумингом:\n", crosstable_inter_proc[crosstable_inter_proc[True] > 40])




data['Predicted Churn'] = ((data['Customer service calls'] > 2) | (data['International plan'] == 'Yes')).astype(bool)

false_negatives = ((data['Churn'] == True) & (data['Predicted Churn'] == False)).sum()
false_positives = ((data['Churn'] == False) & (data['Predicted Churn'] == True)).sum()

total_negatives = (data['Churn'] == True).sum()
total_positives = (data['Churn'] == False).sum()

fn_rate = false_negatives / total_negatives * 100 if total_negatives > 0 else 0
fp_rate = false_positives / total_positives * 100 if total_positives > 0 else 0

print("Процент ложноположительных ошибок:", fp_rate, "%")
print("Процент ложноотрицательных ошибок:", fn_rate, "%")
