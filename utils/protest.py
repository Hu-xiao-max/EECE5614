import numpy as np



actions = [0, 1, 2, 3]
prob_list = [0.7, 0.1, 0.1, 0.1]
# Probility
for _ in range(100):    
    sampled_action = np.random.choice(actions, p=prob_list)
    print(sampled_action)