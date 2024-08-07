# DQN-Module 

## Function name <br>
**dqn_function()** <br>

## Function Code
Copy everything from the file including functions *flatten(lst)* and *unflatten(flat_list, structure)* <br>
**DQN_function.py** <br>

## Function Usage
Has been shown in: <br>
**DQN_MODULE.ipynb**

## Function Parameters
Requires 10 function Parameters: <br>
**mab_function(parameter_values, bounds, m_iterations, e_factor)** <br>
where, <br>
*parameter_values* = A list of all the **parameter values to be optimized** [List, length - Number of parameters] <br>
*bounds* = A list of limit boundaries (range) of the parameter values [List, length - Number of parameters] <br>
*m_iterations* = Number of max iterations [Integer] <br>
*batch_size* = Batch size [Integer] <br>
*memory_size* = Replay memory buffer size [Integer] <br>
*gamma* = Discount rate [Float] <br>
*epsilon* = Exploration rate [Float] <br>
*epsilon_min* = Minimum exploration rate constraint [Float] <br>
*epsilon_decay* = Rate of exploration reduction [Float] <br>
*learning_rate* = Learning rate [Float] <br>

## Function Necessities
To use the dqn function you need to have 1 function: <br>
*1. objective_function(para)* <br>
Takes *para* as function parameter, (expects a list same as *parameter_values*) <br>
Use this function to calculate the sum rate/other value<br>
Return the sum rate/other value [Float] <br>

### Note
Generally with DQN Algorithm, the penalty function differs from problem to problem. You can incorporate the penalty function in the *objective_function(para)* function and return (sum rate + penalty value). <br>
> **Return: Rₛᵤₘ + Total penalty** <br>
