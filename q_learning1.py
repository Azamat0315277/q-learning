import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

env = gym.make("MountainCar-v0", render_mode="human")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * 2  # MountainCar has 2 observation dimensions
obs_space: Box = env.observation_space  # type: ignore
action_space: Discrete = env.action_space  # type: ignore
discrete_os_win_size = (obs_space.high - obs_space.low) / DISCRETE_OS_SIZE
size_q_table = DISCRETE_OS_SIZE + [action_space.n]
q_table = np.random.uniform(low=-2, high=0, size=size_q_table)

epsilon = 0.5  # Starting with balanced exploration/exploitation
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def get_discrete_state(state):
    discrete_state = (state - obs_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"On episode {episode}")
        render = True
    else:
        render = False
    
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    truncated = False
    
    while not (done or truncated):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]).item()
        else:
            action = np.random.randint(0, action_space.n).item()
            
        new_state, reward, done, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()
            
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= 0.5:  # MountainCar goal position
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0
            
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
