# ppo_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import json
from datetime import datetime

# === Configuration ===
MODEL_DIR = "models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class SaveTrajectoryCallback(BaseCallback):
   
    def __init__(self, 
                 eval_env, 
                 save_freq=1000, 
                 log_dir=LOG_DIR, 
                 verbose=1):

        super(SaveTrajectoryCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
    
    def _on_step(self):

        if self.n_calls % self.save_freq == 0:
            # Save current model
            model_path = os.path.join(
                self.log_dir, 
                f"ppo_model_{self.n_calls}_steps.zip"
            )
            self.model.save(model_path)
            
            # Evaluate and save trajectories
            mean_reward = self._evaluate_and_save_trajectories()
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.log_dir, "best_model.zip")
                self.model.save(best_model_path)
                
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")
        
        return True
    
    def _evaluate_and_save_trajectories(self, n_eval_episodes=5):

        all_rewards = []
        
        for episode in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            # Initialize trajectory storage
            trajectory = []
            
            while not (done or truncated):
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                # Store trajectory point
                trajectory.append({
                    'position': {
                        'lat': float(info['current_position']['latitude']),
                        'lon': float(info['current_position']['longitude']),
                        'alt': float(info['current_position']['baro_altitude'])
                    },
                    'turbulence': float(info['turbulence']),
                    'fuel_used': float(info['fuel_used']),
                    'distance_to_destination': float(info['distance_to_destination'])
                })
                
                episode_reward += reward
                episode_length += 1
            
            # Save trajectory to file
            trajectory_path = os.path.join(
                self.log_dir, 
                f"trajectory_step_{self.n_calls}_ep_{episode}.json"
            )
            
            with open(trajectory_path, 'w') as f:
                json.dump({
                    'steps': self.n_calls,
                    'episode': episode,
                    'reward': float(episode_reward),
                    'length': episode_length,
                    'trajectory': trajectory,
                    'total_turbulence': float(info['total_turbulence']),
                    'total_fuel': float(info['total_fuel']),
                    'reached_destination': bool(info['reached_destination'])
                }, f, indent=2)
            
            all_rewards.append(episode_reward)
            
            if self.verbose > 0:
                print(f"Eval episode {episode}: {episode_reward:.2f} reward")
        
        mean_reward = np.mean(all_rewards)
        return mean_reward

def train_flight_route_optimizer(
    env_fn, 
    total_timesteps=100000,
    n_envs=4,
    lr=0.0003,
    save_freq=10000,
    log_dir=LOG_DIR,
    model_dir=MODEL_DIR,
    load_model=None
):

    print(f"Starting PPO training with {total_timesteps} timesteps, {n_envs} environments")
    
    # Create vectorized environment for training
    vec_env = DummyVecEnv([env_fn for _ in range(n_envs)])
    
    # Create a separate environment for evaluation
    eval_env = Monitor(env_fn())
    
    # Neural network policy configuration
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(
            pi=[128, 128],  # Actor network
            vf=[128, 128]   # Critic network
        )
    )
    
    # Create or load model
    if load_model and os.path.exists(load_model):
        print(f"Loading model from {load_model}")
        model = PPO.load(load_model, vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1
        )
    
    # Set up callbacks
    eval_callback = SaveTrajectoryCallback(
        eval_env=eval_env,
        save_freq=save_freq,
        log_dir=log_dir,
        verbose=1
    )
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Training interrupted: {e}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model

def generate_synthetic_test_routes(num_routes=5):

    # Heathrow area boundaries
    LAT_MIN, LAT_MAX = 51.0, 51.6
    LON_MIN, LON_MAX = -0.6, 0.2
    
    routes = []
    
    for i in range(num_routes):
        # Generate random start and destination points
        start_lat = LAT_MIN + 0.1 + np.random.random() * (LAT_MAX - LAT_MIN - 0.2)
        start_lon = LON_MIN + 0.1 + np.random.random() * (LON_MAX - LON_MIN - 0.2)
        
        dest_lat = LAT_MIN + 0.1 + np.random.random() * (LAT_MAX - LAT_MIN - 0.2)
        dest_lon = LON_MIN + 0.1 + np.random.random() * (LON_MAX - LON_MIN - 0.2)
        
        # Ensure start and destination are sufficiently far apart
        while abs(start_lat - dest_lat) + abs(start_lon - dest_lon) < 0.2:
            dest_lat = LAT_MIN + 0.1 + np.random.random() * (LAT_MAX - LAT_MIN - 0.2)
            dest_lon = LON_MIN + 0.1 + np.random.random() * (LON_MAX - LON_MIN - 0.2)
        
        route = {
            'id': f"route_{i+1}",
            'start': {
                'latitude': start_lat,
                'longitude': start_lon,
                'baro_altitude': 8000 + np.random.randint(0, 4000),
                'velocity': 150 + np.random.randint(0, 100),
                'true_track': np.random.randint(0, 360),
                'vertical_rate': 0
            },
            'destination': {
                'latitude': dest_lat,
                'longitude': dest_lon
            }
        }
        
        routes.append(route)
    
    return routes

def evaluate_model_on_routes(
    model, 
    env_fn, 
    routes, 
    save_dir=None
):

    if save_dir is None:
        save_dir = os.path.join(LOG_DIR, "route_evaluations")
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    for i, route in enumerate(routes):
        print(f"Evaluating route {i+1}/{len(routes)}")
        
        # Create environment for this route
        env = env_fn(
            start_point=route['start'],
            destination=route['destination']
        )
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # Run episode
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        # Get trajectory
        trajectory_file = os.path.join(save_dir, f"route_{i+1}_trajectory.json")
        env.save_trajectory(trajectory_file)
        
        # Collect metrics
        route_result = {
            'route_id': route['id'],
            'total_reward': float(total_reward),
            'steps': step_count,
            'total_turbulence': float(info['total_turbulence']),
            'total_fuel': float(info['total_fuel']),
            'reached_destination': bool(info['reached_destination']),
            'trajectory_file': trajectory_file
        }
        
        results.append(route_result)
        
        print(f"Route {i+1}: Reward={total_reward:.2f}, Steps={step_count}, "
              f"Reached={info['reached_destination']}")
    
    # Save summary
    summary_file = os.path.join(save_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation summary saved to {summary_file}")
    
    return results

# Example usage script
if __name__ == "__main__":
    from flight_route_env import FlightRouteEnv, create_synthetic_weather_grid
    
    def make_test_env():

        # Create synthetic weather grid
        weather_grid = create_synthetic_weather_grid()
        
        # Define start and destination
        start_point = {
            'latitude': 51.1,
            'longitude': -0.5,
            'baro_altitude': 5000,
            'velocity': 200,
            'true_track': 90,
            'vertical_rate': 0
        }
        
        destination = {
            'latitude': 51.5,
            'longitude': 0.1
        }
        
        # Create environment
        try:
            model_path = os.path.join("models", "turbulence_model.joblib")
            scaler_path = os.path.join("models", "turbulence_scaler.joblib")
            
            env = FlightRouteEnv(
                turbulence_model_path=model_path,
                scaler_path=scaler_path,
                weather_grid=weather_grid,
                start_point=start_point,
                destination=destination
            )
            return env
        except FileNotFoundError:
            print("Turbulence model not found. Train the model first.")
            return None
    
    # Test environment
    test_env = make_test_env()
    if test_env is not None:
        print("Environment created successfully, ready for training.")
        print("Run 'from main import main; main()' to start the full pipeline.")