import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import gymnasium as gym
import sys
sys.modules['gym'] = gym
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

# Custom Environment for Flight Route Optimization:
class FlightRouteEnv(py_environment.PyEnvironment):
    
    def __init__(self):
        # Define the observation and action specifications
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')
        # Initial state: [latitude, longitude, altitude, velocity, weather_cost]
        self._state = np.array([51.47, -0.45, 3000.0, 140.0, 10.0], dtype=np.float32)
        self._episode_ended = False

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._state = np.array([51.47, -0.45, 3000.0, 140.0, 10.0], dtype=np.float32)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        # Update the state with a simplistic simulation
        self._state[0] += action[0] * 0.01  # Adjust latitude
        self._state[1] += action[0] * 0.01  # Adjust longitude
        self._state[2] += action[1] * 10    # Adjust altitude
        
        # Define a target state for reward calculation (example values)
        target_state = np.array([51.50, -0.40, 3200.0, 140.0, 5.0], dtype=np.float32)
        reward = -np.linalg.norm(self._state - target_state)
        
        # End the episode if deviation is too high
        if np.linalg.norm(self._state - target_state) > 20:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

# Create the Python and TensorFlow environments
py_env = FlightRouteEnv()
tf_env = tf_py_environment.TFPyEnvironment(py_env)

# Define the actor network (policy network)
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(64, 64)
)

# Define the value network (critic)
value_net = value_network.ValueNetwork(
    tf_env.observation_spec(),
    fc_layer_params=(64, 64)
)

# Set up the optimizer and global step counter
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
global_step = tf.compat.v1.train.get_or_create_global_step()

# Initialize the PPO agent
agent = ppo_agent.PPOAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=10,
    train_step_counter=global_step
)
agent.initialize()

# Set up a replay buffer for training
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=1000
)

# Initial data collection using the agent's collect policy
time_step = tf_env.reset()
while not time_step.is_last():
    action_step = agent.collect_policy.action(time_step)
    next_time_step = tf_env.step(action_step.action)
    
    # Fixed: Create trajectory correctly using trajectory.from_transition
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    time_step = next_time_step

# Create a dataset from the replay buffer for training
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    single_deterministic_pass=False
)
iterator = iter(dataset)

# Training loop
num_iterations = 200
for iteration in range(num_iterations):
    time_step = tf_env.current_time_step()
    action_step = agent.collect_policy.action(time_step)
    next_time_step = tf_env.step(action_step.action)
    
    # Fixed: Create trajectory correctly using trajectory.from_transition
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: loss = {train_loss.numpy():.3f}")

# Save the trained model for later inference
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Use TensorFlows checkpoint mechanism instead of agent.save()
checkpoint = tf.train.Checkpoint(agent=agent)
checkpoint_path = checkpoint.save(os.path.join(MODEL_DIR, "ckpt"))
print(f"Checkpoint saved at {checkpoint_path}")
print("Training complete and model saved.")
