import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from helpers import BlobEnv
from collections import deque
import numpy as np
import time
import random
import os
from typing import List
from tqdm import tqdm

REPLAY_MEMORY_SIZE: int = 50_000
MIN_REPLAY_MEMORY_SIZE: int = 1_000
MODEL_NAME: str = "2x256"
DISCOUNT: float = 0.99
MINIBATCH_SIZE: int = 64
UPDATE_TARGET_EVERY: int = 5
MIN_REWARD: int = -200
MEMORY_FRACTION: float = 0.20

EPISODES: int = 20_000
epsilon: float = 1
EPSILON_DECAY: float = 0.9975
MIN_EPSILON: float = 0.001

AGGREGATE_STATS_EVERY: int = 50
SHOW_EVERY: bool = False

env: BlobEnv = BlobEnv()

class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step: int = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._train_dir = self.log_dir
        self._should_write_train_graph = False

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()
        self.step += 1

ep_rewards: List[int] = [-200]

# For more repetitive results
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class DQNAgent:
    def __init__(self):
        
        self.model = self.create_model()
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        
    def create_model(self):
        model = Sequential([
            Conv2D(64,(3,3), input_shape=env.OBSERVATION_SPACE_VALUES),
            Activation("relu"),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            
            Conv2D(64,(3,3)),
            Activation("relu"),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            
            Flatten(),
            Dense(64),
            Dense(env.ACTION_SPACE_SIZE, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MIN_REPLAY_MEMORY_SIZE)
        
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X)/255, 
                       np.array(y), 
                       batch_size=MINIBATCH_SIZE, 
                       verbose=0, 
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
agent: DQNAgent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):
    agent.tensorboard.step = episode
    
    episode_reward = 0
    step = 0
    current_state = env.reset()
    done = False
    
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            
        new_state, reward, done = env.step(action)
        
        episode_reward += reward
        
        if SHOW_EVERY and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        
        current_state = new_state
        step += 1
        
    
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        
        
        
            
