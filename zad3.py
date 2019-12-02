import matplotlib.pyplot as plt
import numpy as np
import collections


array = np.arange(21)

randomArray = np.random.randint(low=0, high=20, size=100)
amountOfValue = []
#print(array)

amount = collections.Counter(randomArray)
#print(amount)
#print(amount[4])

for i in range(len(array)):
    amountOfValue.append(amount[i])

print(array)
print(amountOfValue)

plt.bar(array, amountOfValue)
plt.ylabel('wystapienia')
plt.xlabel('liczby')
plt.suptitle('wykres')
plt.show()

