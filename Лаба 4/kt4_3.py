import pandas as pd

df = pd.read_csv('telecom_churn.csv')

print(df.info())

churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100
print(f"Количество активных клиентов: {churn_counts[False]}")
print(f"Количество потерянных клиентов: {churn_counts[True]}")
print(f"Процент активных клиентов: {churn_percent[False]:.2f}%")
print(f"Процент потерянных клиентов: {churn_percent[True]:.2f}%")

total_minutes = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes']
total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls']
df['Avg call duration'] = total_minutes / total_calls
sorted_df = df.sort_values(by='Avg call duration', ascending=False)
print(sorted_df.head(10))

avg_call_duration_by_churn = df.groupby('Churn')['Avg call duration'].mean()
print(avg_call_duration_by_churn)

avg_service_calls_by_churn = df.groupby('Churn')['Customer service calls'].mean()
print(avg_service_calls_by_churn)

crosstab_churn_service_calls = pd.crosstab(df['Customer service calls'], df['Churn'])
print(crosstab_churn_service_calls)
churn_percentage_by_service_calls = crosstab_churn_service_calls.div(crosstab_churn_service_calls.sum(axis=1),
                                                                     axis=0) * 100
print(churn_percentage_by_service_calls)
high_churn_threshold = churn_percentage_by_service_calls[churn_percentage_by_service_calls[True] > 40]
print(high_churn_threshold)

crosstab_churn_international_plan = pd.crosstab(df['International plan'], df['Churn'])
print(crosstab_churn_international_plan)
churn_percentage_by_international_plan = crosstab_churn_international_plan.div(
    crosstab_churn_international_plan.sum(axis=1), axis=0) * 100
print(churn_percentage_by_international_plan)

df['Predicted Churn'] = (df['Customer service calls'] > 3) | (df['International plan'] == 'Yes')
comparison = df[['Churn', 'Predicted Churn']]
comparison['Correct'] = comparison['Churn'] == comparison['Predicted Churn']
false_positives = comparison[(comparison['Churn'] == False) & (comparison['Predicted Churn'] == True)].shape[0]
false_negatives = comparison[(comparison['Churn'] == True) & (comparison['Predicted Churn'] == False)].shape[0]
total_predictions = comparison.shape[0]
print(f"Процент ложноположительных ошибок: {false_positives / total_predictions * 100:.2f}%")
print(f"Процент ложноотрицательных ошибок: {false_negatives / total_predictions * 100:.2f}%")
