import sys
import pickle
import matplotlib.pyplot as plt

n = int(sys.argv[2])

arr = []

all_train = []
all_test = []

for i in range(n):
    file_name = sys.argv[1] + "acc_" + str(i)
    data = pickle.load(open(file_name, "rb"))
    arr.append(data)

    all_train += data['train_acc']
    all_test += data['test_acc']

plt.plot(all_train, color='red')
plt.plot(all_test, color='green')
plt.show()




