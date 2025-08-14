from time import sleep
import re

def check_name(name, names):
    if name in names:
        return True
    else:
        print("Ошибка. Такого имени нет в списке.")
        return False
    
def check_sum(sum):
    if sum.isdigit():
        return True        
    else:
        print("Ошибка чтения суммы. Введите число")
        return False

while True:
    check_all_names = True
    names = input("Введите имена всех участников похода через пробел: ").split(" ")
    if len(names) < 2:
        print("Ошибка. Введите больше имен.")
        continue
    for name in names:
        if re.match("^[A-Za-z]*$", name):
            continue
        else:
            check_all_names = False
            print('Ошибка чтения имени "'+ name+'"')
            sleep(0.5)
    if check_all_names:
        break        

names = list(set(names))
data = {name : 0 for name in names}

while True:
    n = input("Введите количество покупок: ")
    if n.isdigit():
        break
    else:
        print("Введите положительное число\n")
        sleep(1.5)
        continue

names = ['I', 'A', 'Ig']
#data = {'I':700, 'A':100, 'Ig': 0}
print(names)
n = int(n)
for i in range(n):
    
    while True:
        flag_name = True
        flag_sum = True
        operation = input("Введите имя человека и сумму покупки через пробел: ").split(" ")

        #print(operation)

        if len(operation) < 2:
            print("Ошибка. Неверное количество аргументов.")
            flag_name = False
            flag_sum = False
            continue

        flag_name = check_name(operation[0], names)
        flag_sum = check_sum(operation[1])


        if flag_sum and flag_name:
            data[operation[0]] += int(operation[1])
            break
        else:
            continue

dolzhniki = []
ne_dolzhniki = []


sum_for_one = round(sum(data.values())/len(names), 2)

for name, trata in data.items():
    balance = round(trata - sum_for_one, 2)

    if balance >= 0:
        ne_dolzhniki.append((name, balance))
    else:
        dolzhniki.append((name, balance * -1))

ans = []

i = 0
j = 0
#print(ne_dolzhniki)
#print(dolzhniki)
while i < len(dolzhniki) and j < len(ne_dolzhniki):
    
    dolzhnik_name, dolg = dolzhniki[i]
    ne_dolzhnik_name, ostatok = ne_dolzhniki[j]
    
    perevod = min(dolg, ostatok)

    ans.append((dolzhnik_name, ne_dolzhnik_name, perevod))

    dolzhniki[i] = (dolzhnik_name, round(dolg-perevod, 2))
    ne_dolzhniki[j] = (ne_dolzhnik_name, round(ostatok - perevod, 2))

    if dolzhniki[i][1] == 0:
        i += 1
    if ne_dolzhniki[j][1] == 0:
        j += 1

print("Всего вам понадобится", len(ans), "переводов")
for a in ans:
    print("От", a[0], "к", a[1], ' на сумму:', a[2])




