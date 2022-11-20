import threading

def tt():
    for i in range(10):
        print('AAAAA', i)
x = threading.Thread(target=tt)
x.start()
for i in range(7):
        print('B', i)