
from datetime import datetime


pre_time = datetime.now()

x = 0

for i in range(100000):
    x = x*i + i

now_time = datetime.now()

print((now_time-pre_time).total_seconds())