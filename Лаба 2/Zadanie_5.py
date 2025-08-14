import time

#Входные данные: 
#Первая строка - количество записей в блокноте
#Следующие n строк - записи в формате ( дата, название пиццы, стоимость заказа )
def read_data(key):
    data = {}
    while True:
        n = input("Введите количество записей: ")
        if n.isdigit():
            n = int(n)
            break
        else:
            print("Введите положительное число\n")
            continue
    
    for i in range(n):

        while True:
            notice = input().split()
            if len(notice) < 3:
                print("Ошибка. Неверный формат. Нужно в формате: дата(день.месяц.год), название пиццы, стоимость заказа")
            else:
                try:
                    notice[2] = int(notice[2])
                except:
                    print("Ошибка. Неверный формат. Нужно в формате: дата(день.месяц.год), название пицыы, стоимость заказа")
                    continue
                if notice[0].count('.') != 2:
                    print("Формат записи даты неверный. Нужно в формате: день.месяц.год")
                    continue


                break
        
       
        
        if notice[key] in data:
            data[notice[key]] += [notice]
        else:
            data[notice[key]] = [notice]

    return data
def first():
    data = read_data(1)
    for name, costs in data.items():
        data[name] = len(costs)
    data = sorted(data.items(), key=lambda item: item[1], reverse=True)
    print("Запрос выполнен\n")
    for info in data:
        print(info[0], '-', info[1], 'продано')

def second():
    data = read_data(0)
    
    for date, info in data.items():
        summ = 0
        for zakaz in info:
            summ += zakaz[2]
        print("Дата:", date, "- Общая сумма:", summ)
    


def third():
    data = read_data(2)
    data = sorted(data.items(), reverse=True)
    summ, info = data[0]

    for i in info:
        print("Сумма заказа:", summ)
        print("Название -", i[1])
        print("Дата заказа:", i[0]) 

def fouth():
    while True:
        n = input("Введите количество записей: ")
        if n.isdigit():
            n = int(n)
            break
        else:
            print("Введите положительное число\n")
            continue
    
    ans = []
    for i in range(n):
        notice = input().split()
        ans.append(int(notice[2]))
        
    print("Средняя стоимость заказа:", sum(ans)/len(ans))




while True:
    print("Выберите информацию, которую хотите узнать:\n1. Список всех заказанных пицц и их количество\n2. Список всех дат и сумма проданных в этот день пицц\n3. Информация о самом дорогом заказе\n4. Средняя стоимость заказа")
    while True:
        choise = input("Введите вариант: ")
        if choise.isdigit():
            break
        else:
            print("Введите положительное число\n")
            time.sleep(1.5)
            continue
            


    if choise == '1':
        first()
        
    elif choise == '2':
        second()
    
    elif choise == '3':
        third()
        
    elif choise == '4':
        fouth()
    else:
        print("Такой опции нет")
        time.sleep(1.5)
