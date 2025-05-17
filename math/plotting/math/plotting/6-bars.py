#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # apples, bananas, oranges, peaches
labels = ['apples', 'bananas', 'oranges', 'peaches']

x = np.arange(fruit.shape[1])
bottom = np.zeros(fruit.shape[1])

for i in range(fruit.shape[0]):
    plt.bar(x, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=labels[i])
    bottom += fruit[i]

plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend()
plt.ylim(0, 80)

plt.show()
