import numpy as np

countries = np.genfromtxt("Лаба 3\global-electricity-generation.csv", dtype = "U20", delimiter = ",", skip_header=1)[:,0]
generation = np.genfromtxt("Лаба 3\global-electricity-generation.csv", delimiter = ",", skip_header=1)[:,1:]
consumption = np.genfromtxt("Лаба 3\global-electricity-consumption.csv", delimiter = ",", skip_header=1)[:,1:]

#2 задание
avg_gen_last5 = np.nanmean(generation[:, -5:], axis=1)
print("Среднее  стран произовдство за последние 5 лет: \n", avg_gen_last5)
avg_con_last5 = np.nanmean(consumption[:, -5:], axis=1)
print("Среднее потребление стран за последние 5 лет: \n", avg_con_last5)
#3.1
sum_by_country_every_year = np.nansum(consumption, axis = 0)
print("Суммарное потребление электроэнегрии по странам: \n", sum_by_country_every_year)
#3.2
max_gen_per_year = np.nanmax(generation)
print("Максимальное кол-во электроэнегрии за один год: \n", max_gen_per_year)
#3.3
high_gen_countries = countries[avg_gen_last5 > 500]
print("Списое стран с производством свыше 500 млдр. кВт*ч: \n", high_gen_countries)
#3.4
quantile_90 = np.percentile(avg_con_last5, 90)
top_10_countries_by_con = countries[avg_con_last5 > quantile_90]
print("Топ 10% стран, которые потребляют больше всего энергии: \n", top_10_countries_by_con)
#3.5
lvl_up_since_1992 = countries[generation[:, -1] / generation[:, 0] > 10]
print("Страны, увеличившие производство с 1992 более чем в 10 раз: \n", lvl_up_since_1992)
#3.6
total_con_by_county = np.nansum(consumption, axis = 1)
total_gen_by_county = np.nansum(generation, axis = 1)
countries_with_higher_con = countries[(total_con_by_county >100) & (total_con_by_county > total_gen_by_county)]
print("Cтраны, потратившие за все годы больше 100 млрд. кВт*ч электроэнергии и при этом произвели меньше, чем потратили: \n", countries_with_higher_con)
#3.7
country_biggest_con_2020 = countries[np.argmax(consumption[:, -2])]
print("Потратили больше всего энергии в 2020:", country_biggest_con_2020)
