# api/optimize_flight.py

import os
import json
from flask import Flask, request, jsonify
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
from tf_agents.utils import common

# Rebuild the custom FlightRouteEnv:

class FlightRouteEnv(py_environment.PyEnvironment):

    def __init__(self):
        # Define observation and action specs
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')
       
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
        
        # Update state 
        self._state[0] += action[0] * 0.01  # Adjust latitude
        self._state[1] += action[0] * 0.01  # Adjust longitude
        self._state[2] += action[1] * 10    # Adjust altitude

        # Example target for reward calculation
        target_state = np.array([51.50, -0.40, 3200.0, 140.0, 5.0], dtype=np.float32)
        reward = -np.linalg.norm(self._state - target_state)

        # End the episode if deviation is too high
        if np.linalg.norm(self._state - target_state) > 20:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

# Recreate the TF environment and agent:

py_env = FlightRouteEnv()
tf_env = tf_py_environment.TFPyEnvironment(py_env)

# Create the actor (policy) network
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(64, 64)
)

# Create the value (critic) network
value_net = value_network.ValueNetwork(
    tf_env.observation_spec(),
    fc_layer_params=(64, 64)
)

# Set up the optimizer and global step
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

# Load the trained checkpoint:

MODEL_DIR = "models"
checkpoint = tf.train.Checkpoint(agent=agent)
latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
if latest_ckpt:
    checkpoint.restore(latest_ckpt).assert_existing_objects_matched()
    print(f"Checkpoint restored from {latest_ckpt}")
else:
    print("No checkpoint found. Ensure you have trained and saved the model.")


# Create the Flask API: 

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize_flight():
    data = request.get_json()
    if not data or 'state' not in data:
        return jsonify({'error': 'Invalid input, expected JSON with "state" key.'}), 400
    
    # Convert the input to a numpy array 
    state = np.array(data['state'], dtype=np.float32)
    
    # Create a time step from the state 
    time_step = ts.restart(state)
    
    # Get action from the agentts policy
    action_step = agent.policy.action(time_step)
    action = action_step.action.numpy().tolist()
    
    # Optionally, simulate one step to see the next state
    next_time_step = tf_env.step(action_step.action)
    next_state = next_time_step.observation.numpy().tolist()
    
    return jsonify({
        'action': action,
        'next_state': next_state
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
