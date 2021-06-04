import time
import dis

n = 10000
a = [1 for i in range(n)]

time1 = time.time()
def t1():
    for i in range(n):
        a[i] = -a[i]
print(time.time()-time1)

time1 = time.time()
def t2():
    for i in range(n):
        a[i] *= -1
print(time.time()-time1)


time1 = time.time()
a = [-i for i in a]
print(time.time()-time1)

dis.dis(t1)
print('---------------')
dis.dis(t2)