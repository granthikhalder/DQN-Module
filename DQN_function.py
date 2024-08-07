import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def unflatten(flat_list, structure):
    flat_iter = iter(flat_list)
    def helper(struct):
        result = []
        for elem in struct:
            if isinstance(elem, list):
                result.append(helper(elem))
            else:
                result.append(next(flat_iter))
        return result
    return helper(structure)

def build_model(s_size, a_size):
    model = nn.Sequential(
        nn.Linear(s_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, a_size)
    )
    return model

def dqn_function(parameter_values, bounds, m_iterations, batch_size, memory_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate):

    print("DQN Algorithm Started")

    para = flatten(parameter_values)
    len_para = len(para)

    max_iterations = m_iterations
    memory = deque(maxlen = memory_size)
    reward_array = []
    state_size = len_para
    action_size = 2 * len_para

    model = build_model(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    initial_values = [bounds[i][0] for i in range(len_para)]
    state = initial_values

    def remember(state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))

    def act(state):
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = model(state)
        return torch.argmax(act_values).item()

    def replay(batch_size):
        nonlocal epsilon
        if len(memory) < batch_size:
            return
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + gamma * torch.max(model(next_state)).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = model(state).detach().clone()
            target_f[0][action] = torch.tensor(target)
            model.zero_grad()
            loss = criterion(model(state), target_f)
            loss.backward()
            optimizer.step()
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

    for iter in range(max_iterations):

        action = act(state)

        if(action < len_para):
            para[action] = max(bounds[action][0], min(bounds[action][1], para[action] + np.random.uniform(-0.1, 0)))
        elif(action >= len_para):
            temp_action = action - len_para
            para[temp_action] = max(bounds[temp_action][0], min(bounds[temp_action][1], para[temp_action] + np.random.uniform(0, 0.1)))

        temp = unflatten(para, parameter_values)
        reward = objective_function(temp)

        next_state = para.copy()
        done = iter == max_iterations - 1

        remember(state, action, reward, next_state, done)
        state = next_state.copy()

        if len(memory) > batch_size:
            replay(batch_size)

        reward_array.append(reward)

        if iter % 100 == 0 or iter == (max_iterations-1):
            print(f"Iteration {iter}: Best value = {max(reward_array)}")
    return reward_array