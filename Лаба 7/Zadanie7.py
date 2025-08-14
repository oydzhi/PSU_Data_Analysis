import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def calculate_stats(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return f1, precision, recall



data = pd.read_csv('Laba7.csv', delimiter=';')




data.drop(columns=['УИД_Брони'], inplace=True)
data.drop(columns=['ВремяБрони'], inplace=True)
data.drop(columns=['ДатаБрони'], inplace=True)

data['ПродаваемаяПлощадь'] = data['ПродаваемаяПлощадь'].str.replace(',', '.').astype(float)
data['СтоимостьНаДатуБрони'] = data['СтоимостьНаДатуБрони'].str.replace(',', '.').astype(float)
data['СкидкаНаКвартиру'] = data['СкидкаНаКвартиру'].str.replace(',', '.').astype(float)
data['ФактическаяСтоимостьПомещения'] = data['ФактическаяСтоимостьПомещения'].str.replace(',', '.').astype(float)
data['Тип'] = data['Тип'].str.replace(',', '.')



def convert_rooms(value):
    if isinstance(value, str) and value.endswith('к'):
        return pd.to_numeric(value[:-1], errors='coerce')
    return np.nan

data['Тип'] = data['Тип'].apply(convert_rooms)
#print(data['Тип'])

data = data[data['ВидПомещения'] == 'жилые помещения']
data.drop(columns=['ВидПомещения'], inplace=True)

data = data[data['СледующийСтатус'].isin(['Продана', 'Свободна'])]
data['СледующийСтатус'] = data['СледующийСтатус'].map({'Продана': 1, 'Свободна': 0})

data['ПродаваемаяПлощадь'] = pd.to_numeric(data['ПродаваемаяПлощадь'], errors='coerce')

data['ИсточникБрони'] = data['ИсточникБрони'].map({'МП': 1, 'ручная': 0})
data['ВременнаяБронь'] = data['ВременнаяБронь'].map({'Да': 1, 'Нет': 0})
data['ТипСтоимости'] = data['ТипСтоимости'].map({'Стоимость при 100% оплаты': 1, 'Стоимость в рассрочку': 0})
data['ВариантОплаты'] = data['ВариантОплаты'].map({'Единовременная оплата ': 1, 'Оплата в рассрочку': 0})

data['ВариантОплатыДоп'] = data['ВариантОплатыДоп'].fillna(data['ВариантОплаты'])
data['ВариантОплатыДоп'] = data['ВариантОплатыДоп'].map({'Ипотека': 1, 'Вторичное жилье': 0, 'Единовременная оплата': 0, 'Оплата в рассрочку': 1})

data['СделкаАН'] = data['СделкаАН'].map({'Да': 1, 'Нет': 0})
data['ИнвестиционныйПродукт'] = data['ИнвестиционныйПродукт'].map({'Да': 1, 'Нет': 0})
data['Привилегия'] = data['Привилегия'].map({'Да': 1, 'Нет': 0})

data['Статус лида (из CRM)'] = data['Статус лида (из CRM)'].map({'S': 1, 'P': 0, 'F': -1})

data = pd.get_dummies(data, columns=['Город'])

print(data.head(10))

data['СкидкаНаКвартиру'].fillna(0, inplace=True)
data['Тип'].fillna(data['Тип'].median(), inplace=True)
data['ПродаваемаяПлощадь'].fillna(data['ПродаваемаяПлощадь'].median(), inplace=True)
data['ВариантОплатыДоп'].fillna(data['ВариантОплаты'], inplace=True)

data['Цена за квадратный метр'] = data['ФактическаяСтоимостьПомещения'] / data['ПродаваемаяПлощадь']
data['Скидка в процентах'] = data['СкидкаНаКвартиру'] / data['ФактическаяСтоимостьПомещения']

numeric_columns = ['Тип', 'ПродаваемаяПлощадь', 'Этаж', 'СтоимостьНаДатуБрони', 
                   'СкидкаНаКвартиру', 'ФактическаяСтоимостьПомещения', 'Цена за квадратный метр']

for column in numeric_columns:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot для столбца: {column}')
    plt.xlabel(column)
    plt.show()

scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data)

data['СкидкаНаКвартиру'] = (data['СкидкаНаКвартиру'] - 0.5) / (data['СкидкаНаКвартиру'].max() - data['СкидкаНаКвартиру'].min())

target_counts = data['СледующийСтатус'].value_counts()
print('Сбалансированность датасета:\n', target_counts)

X = data.drop(columns=['СледующийСтатус'])
y = data['СледующийСтатус']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred_knn_train = knn.predict(X_train)
y_pred_knn_test = knn.predict(X_test)

y_pred_tree_train = tree.predict(X_train)
y_pred_tree_test = tree.predict(X_test)

f1_knn_train, precision_knn_train, recall_knn_train = calculate_stats(y_train, y_pred_knn_train)
f1_knn_test, precision_knn_test, recall_knn_test = calculate_stats(y_test, y_pred_knn_test)

f1_tree_train, precision_tree_train, recall_tree_train = calculate_stats(y_train, y_pred_tree_train)
f1_tree_test, precision_tree_test, recall_tree_test = calculate_stats(y_test, y_pred_tree_test)

print(f'КНН - F1 Трен: {f1_knn_train}, F1 Тест: {f1_knn_test}')
print(f'КНН - Precision Трен: {precision_knn_train}, Precision Тест: {precision_knn_test}')
print(f'КНН - Recall Трен: {recall_knn_train}, Recall Тест: {recall_knn_test}')

print(f'Дерево - F1 Трен: {f1_tree_train}, F1 Тест: {f1_tree_test}')
print(f'Дерево - Precision Трен: {precision_tree_train}, Precision Тест: {precision_tree_test}')
print(f'Дерево - Recall Трен: {recall_tree_train}, Recall Тест: {recall_tree_test}')



f1_scores_knn = []

for k in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=k)
    f1_scores_knn.append(np.mean(cross_val_score(knn, X_train, y_train, cv=5, scoring='f1')))

plt.plot(range(1, 41), f1_scores_knn)
plt.xlabel('Количество соседей k')
plt.ylabel('F1 Score')
plt.title('Зависимость F1 от количества соседей k для KNN')
plt.show()

f1_scores_tree = []

for depth in range(2, 41):
    tree = DecisionTreeClassifier(max_depth=depth, class_weight='balanced')
    f1_scores_tree.append(np.mean(cross_val_score(tree, X_train, y_train, cv=5, scoring='f1')))

plt.plot(range(2, 41), f1_scores_tree)
plt.xlabel('Глубина дерева')
plt.ylabel('F1 Score')
plt.title('Зависимость F1 от глубины дерева для Decision Tree')
plt.show()

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

f1_log_reg = f1_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)

print(f'Логистическая регрессия - F1: {f1_log_reg}')
print(f'Логистическая регрессия - Precision: {precision_log_reg}')
print(f'Логистическая регрессия - Recall: {recall_log_reg}')

svm = LinearSVC(max_iter=1000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

f1_svm = f1_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)

print(f'SVM - F1: {f1_svm}')
print(f'SVM - Precision: {precision_svm}')
print(f'SVM - Recall: {recall_svm}')