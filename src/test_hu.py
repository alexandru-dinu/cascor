import sys
import pickle
import matplotlib.pyplot as plt

name = sys.argv[1]

data = pickle.load(open(name, "rb"))

print(data)

plt.plot(data['train'], color='red')
plt.plot(data['test'], color='green')
plt.show()
