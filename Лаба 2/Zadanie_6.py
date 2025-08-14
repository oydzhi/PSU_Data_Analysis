
import requests
mbox = requests.get('http://www.py4inf.com/code/mbox.txt').text
data = mbox.split('\n')
print(123)


ans = {}

for stroka in data:
    if stroka.find("From ") == 0:
        author = stroka.split(" ")[1]
        if author in ans:
            ans[author] += 1
        else:
            ans[author] = 1

print("Больше всего сообщений от:", sorted(ans.items(), key=lambda item: item[1], reverse=True)[0][0])
