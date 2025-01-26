#
# # Upper Conficence Bound (UCB) is a reinforcement learning algorithem.
# 


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Upper Conficence Bound (UCB)
import math
total_numberRounds = 10000
total_numberAdvertisements = 10
advertisements_selected = []
numbers_of_selections = [0] * total_numberAdvertisements
sums_of_rewards = [0] * total_numberAdvertisements
total_reward = 0
for number_round in range(0, total_numberRounds):
    advertisement_index = 0
    max_upperBound = 0
    for counter_i in range(0, total_numberAdvertisements):
        if (numbers_of_selections[counter_i] > 0):
            average_reward = sums_of_rewards[counter_i] / numbers_of_selections[counter_i]
            delta_i = math.sqrt(3/2 * math.log(number_round + 1) / numbers_of_selections[counter_i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upperBound:
            max_upperBound = upper_bound
            advertisement_index = counter_i
    advertisements_selected.append(advertisement_index)
    numbers_of_selections[advertisement_index] = numbers_of_selections[advertisement_index] + 1
    reward = dataset.values[number_round, advertisement_index]
    sums_of_rewards[advertisement_index] = sums_of_rewards[advertisement_index] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(advertisements_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print("Hello")