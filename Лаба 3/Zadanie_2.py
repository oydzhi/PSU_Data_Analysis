from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("data2.csv", delimiter = ";")

x = data[:, 0]
f_x = data[:, 1]

x[0] = 0


x_dots_2 = np.array([x[0], x[len(x)//2], x[-1]])
f_x_dots_2 = np.array([f_x[0], f_x[len(x)//2], f_x[-1]])

x_dots_3 = np.array([x[0], x[len(x)//3], x[2*len(x)//3], x[-1]])
f_x_dots_3 = np.array([f_x[0], f_x[len(x)//3], f_x[2*len(x)//3], f_x[-1]])




A2 = np.vstack([x_dots_2**2, x_dots_2, np.ones(len(x_dots_2))]).T
A3 = np.vstack([x_dots_3**3, x_dots_3**2, x_dots_3, np.ones(len(x_dots_3))]).T



coeffs2 = solve(A2.T @ A2, A2.T @ f_x_dots_2)
coeffs3 = solve(A3.T @ A3, A3.T @ f_x_dots_3)

print("Коэффициенты полинома второй степени:", coeffs2)
print("Коэффициенты полинома третьей степени:", coeffs3)

A2 = np.vstack([x**2, x, np.ones(len(x))]).T
A3 = np.vstack([x**3, x**2, x, np.ones(len(x))]).T

f_x_pred2 = A2 @ coeffs2
f_x_pred3 = A3 @ coeffs3

rss2 = np.sum((f_x - f_x_pred2)**2)
rss3 = np.sum((f_x - f_x_pred3) ** 2)

print("RSS для полинома второй степени:", rss2)
print("RSS для полинома третьей степени:", rss3)
if rss2 < rss3:
    print("Полином второй степени лучше.")
else:
    print("Полином третьей степени лучше.")


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x, f_x, color='blue', label='Исходные данные')
plt.plot(x, f_x_pred2, color='red', label='Полином 2-й степени')
plt.xlabel('Скидка')
plt.ylabel('Прибыль')
plt.title('Полином 2-й степени')
plt.legend()

# График полинома 3-й степени
plt.subplot(1, 2, 2)
plt.scatter(x, f_x, color='blue', label='Исходные данные')
plt.plot(x, f_x_pred3, color='green', label='Полином 3-й степени')
plt.xlabel('Скидка')
plt.ylabel('Прибыль')
plt.title('Полином 3-й степени')
plt.legend()

plt.tight_layout()
plt.show()

A, B, C, D = coeffs3
for x in [6, 8]:
    f_x = A * x**3 + B * x**2 + C * x + D
    print("Значение полинома 3-й степени для x=" + str(x) + ":", f_x)

'''Коэффициенты полинома второй степени: [ -1.19601759  13.8664875  -10.92202516]
Коэффициенты полинома третьей степени: [-0.58561475  3.19609302  5.25326578 -7.68943175]
RSS для полинома второй степени: 63.05323469493738
RSS для полинома третьей степени: 1.3911531217917987
Полином третьей степени лучше.
Значение полинома 3-й степени для x=6: 12.396725946075644
Значение полинома 3-й степени для x=8: -60.94810348181406'''
