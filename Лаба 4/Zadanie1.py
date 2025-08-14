import numpy as np
import pandas as pd

M = 1.0
s = 1.0
data = np.random.normal(M, s, 1000)
print(data)
series = pd.Series(data)

prop1 = ((series > (M - s)) & (series < (M + s))).mean()
prop3 = ((series > (M - 3 * s)) & (series < (M + 3 * s))).mean()
#print(prop1, "---", prop3)
t_prop3 = 0.997  

res = abs(prop3 - t_prop3)
print("Теоритическое ожидание -", t_prop3)
print("Фактически - ", prop3)
print("Разница -", res)

series_sqrt = np.sqrt(series)
print(series_sqrt)

mean_sqrt = series_sqrt.mean()
print(mean_sqrt)

dtf = pd.DataFrame({
    'number': series,
    'root': series_sqrt
})

print(dtf.head(6))

res_query = dtf.query('root >= 1.8 and root <= 1.9')
print(res_query)
