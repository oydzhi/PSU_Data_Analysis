import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



data = pd.read_csv('Laba6.csv')

print("Тип данных каждого столбца:\n", data.dtypes)

print("Количество отстутсвующих значений:\n", data.isnull().sum())
data.fillna(data.median(), inplace=True)

cor_matrix = data.corr()
plt.figure(figsize=(12, 8))
plt.title("Корреляционная матрица")
sns.heatmap(cor_matrix, annot=True, fmt=".3f", cmap='coolwarm')#отражает зависимость одной переменной от другой, изменяется от -1 до 1
plt.show()       #1 - увеличивается одна - увеличивается и другая, 0 - зависимости нет, -1 - увеличивается первая - другая уменьшается


the_most_corr = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'TAX', 'NOX']

for priznak in the_most_corr:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[priznak], y=data['MEDV'])
    plt.title('Диаграмма рассеяния между '+ priznak + ' и MEDV')
    plt.xlabel(priznak)
    plt.ylabel('MEDV')
    plt.show()



X = data[['RM', 'LSTAT', 'PTRATIO']]#входные данные(признаки)
y = data['MEDV']#целевые переменные(метки) (здесь переменная одна)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
'''
print(X_train[:5], '-', y_train_pred[:5])
print('\n')
print(X_test[:5], '-', y_test_pred[:5])
''' 




r2_train = round(r2_score(y_train, y_train_pred), 2)
rmse_train = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)

# Оценка для тестовой выборки
r2_test = round(r2_score(y_test, y_test_pred), 2)
rmse_test = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)

print('\nТренировочное R^2:', r2_train, 'Тренировочное RMSE:', rmse_train)
print('Тестовое R^2:', r2_test, 'Тестовое RMSE:', rmse_test, '\n')

#######################################################################################################################################

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['MEDV'])
plt.title('Ящик с "усами" для MEDV')
plt.show()


data_f = data[data['MEDV'] < 38]
X_f = data_f[['RM', 'LSTAT', 'PTRATIO']]
y_f = data_f['MEDV']



X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2)
model.fit(X_train_f, y_train_f)


y_train_pred_f = model.predict(X_train_f)
y_test_pred_f = model.predict(X_test_f)


r2_train_f = round(r2_score(y_train_f, y_train_pred_f), 2)
rmse_train_f = round(np.sqrt(mean_squared_error(y_train_f, y_train_pred_f)), 2)
r2_test_f = round(r2_score(y_test_f, y_test_pred_f), 2)
rmse_test_f = round(np.sqrt(mean_squared_error(y_test_f, y_test_pred_f)), 2)

print('Тренировочный R^2 без выбросов:', r2_train_f, 'Тренировочный RMSE без выбросов:', rmse_train_f)
print('Тестовый R^2 без выбросов:', r2_test_f, 'Тестовый RMSE без выбросов:', rmse_test_f, '\n')





from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_f, y_train_f)


y_train_pred_ridge = ridge_model.predict(X_train_f)
y_test_pred_ridge = ridge_model.predict(X_test_f)


r2_ridge_train = round(r2_score(y_train_f, y_train_pred_ridge), 2)
rmse_ridge_train = round(np.sqrt(mean_squared_error(y_train_f, y_train_pred_ridge)), 2)
r2_ridge_test = round(r2_score(y_test_f, y_test_pred_ridge), 2)
rmse_ridge_test = round(np.sqrt(mean_squared_error(y_test_f, y_test_pred_ridge)), 2)

print('Гребневый тренировочный R^2:', r2_ridge_train, 'Гребневый тренировочный RMSE:', rmse_ridge_train)
print('Гребневый тестовый R^2', r2_ridge_test, 'Гребневый тестовый RMSE:', rmse_ridge_test, '\n')





from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train_f)


poly_model = LinearRegression()
poly_model.fit(X_poly, y_train_f)
X_test_poly = poly.transform(X_test_f)


y_train_pred_poly = poly_model.predict(X_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)


r2_poly_train = round(r2_score(y_train_f, y_train_pred_poly), 2)
rmse_poly_train = round(np.sqrt(mean_squared_error(y_train_f, y_train_pred_poly)), 2)
r2_poly_test = round(r2_score(y_test_f, y_test_pred_poly), 2)
rmse_poly_test = round(np.sqrt(mean_squared_error(y_test_f, y_test_pred_poly)),2)

print('Полиномиальный тренировочный R^2:', r2_poly_train, 'Полиномиальный тренировочный RMSE:', rmse_poly_train)
print('Полиномиальный тестовый R^2:', r2_poly_test, 'Полиномиальный тестовый RMSE:', rmse_poly_test, '\n')