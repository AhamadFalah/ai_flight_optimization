# flight_route_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import joblib
import json
from datetime import datetime
import pandas as pd

# Math utilities
from math import radians, cos, sin, asin, sqrt, atan2, degrees

def create_synthetic_weather_grid(num_rows=10, num_cols=10):
    # Heathrow area boundaries
    LAT_MIN, LAT_MAX = 51.0, 51.6
    LON_MIN, LON_MAX = -0.6, 0.2
    
    lat_step = (LAT_MAX - LAT_MIN) / num_rows
    lon_step = (LON_MAX - LON_MIN) / num_cols
    
    weather_grid = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            lat = LAT_MIN + (i + 0.5) * lat_step
            lon = LON_MIN + (j + 0.5) * lon_step
            
            # Create spatially coherent weather patterns
            weather_point = {
                "lat": lat,
                "lon": lon,
                "temperature": 15 + np.random.uniform(-5, 5),
                "pressure": 1013 + np.random.uniform(-10, 10),
                "humidity": 60 + np.random.uniform(-20, 20),
                "wind_speed": 5 + np.random.uniform(0, 15),
                "wind_deg": np.random.uniform(0, 360)
            }
            
            weather_grid.append(weather_point)
    
    return weather_grid

class FlightRouteEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(
        self, 
        turbulence_model_path, 
        scaler_path, 
        weather_grid=None, 
        start_point=None, 
        destination=None,
        max_steps=200,
        reward_weights=None
    ):

        super(FlightRouteEnv, self).__init__()
        
        # Load turbulence prediction model and scaler
        self.turbulence_model = joblib.load(turbulence_model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Get feature names (if available)
        self.feature_names = getattr(self.scaler, 'feature_names_in_', [
            'velocity', 'baro_altitude', 'vertical_rate', 
            'temperature', 'pressure', 'humidity', 
            'wind_speed', 'wind_deg'
        ])
        
        # Weather and route configuration
        self.weather_grid = weather_grid or self._generate_synthetic_weather_grid()
        
        # Flight route details
        self.start_point = start_point or self._generate_default_start_point()
        self.destination = destination or self._generate_default_destination()
        
        # Environment parameters
        self.max_steps = max_steps
        # NEW: Define a time step (in seconds) for each update; reduced from 300 to 30 for finer simulation steps
        self.time_step = 5
        
        # Reward configuration
        default_weights = {
            'turbulence': 5.0,      # Penalty for turbulence
            'distance': 1.0,        # Reward for reducing distance to destination
            'fuel': 2.0,            # Penalty for fuel usage
            'maneuver': 0.5,        # Penalty for excessive maneuvering
            'completion': 100.0     # Bonus for reaching destination
        }
        self.reward_weights = reward_weights or default_weights
        
        # Geographic boundaries (Heathrow area)
        self.LAT_MIN, self.LAT_MAX = 51.0, 51.6
        self.LON_MIN, self.LON_MAX = -0.6, 0.2
        
        # Action and observation spaces
        self._define_action_space()
        self._define_observation_space()
        
        # Initialize state
        self.reset()
    
    def _define_action_space(self):
        
        # Possible heading changes (degrees)
        self.heading_changes = [-30, -20, -10, 0, 10, 20, 30]
        
        # Possible altitude changes (feet)
        self.altitude_changes = [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000]
        
        # Total number of discrete actions
        self.action_space = spaces.Discrete(
            len(self.heading_changes) * len(self.altitude_changes)
        )
    
    def _define_observation_space(self):

        self.observation_space = spaces.Box(
            low=np.array([
                self.LAT_MIN, self.LON_MIN, 0,     # Position min
                0, 0, -50,                         # Speed, heading, vertical rate min
                -50, 900, 0, 0, 0,                 # Weather min
                0, 0,                              # Distance, bearing min
                0, 0, 0                            # Turbulence ahead min
            ], dtype=np.float32),
            high=np.array([
                self.LAT_MAX, self.LON_MAX, 40000, # Position max
                500, 360, 50,                      # Speed, heading, vertical rate max
                50, 1100, 100, 50, 360,            # Weather max
                100, 360,                          # Distance, bearing max
                1, 1, 1                            # Turbulence ahead max
            ], dtype=np.float32),
            dtype=np.float32
        )

    def _generate_synthetic_weather_grid(self, num_rows=10, num_cols=10):

        return create_synthetic_weather_grid(num_rows, num_cols)
    
    def _generate_default_start_point(self):

        return {
            'latitude': self.LAT_MIN + 0.2 * (self.LAT_MAX - self.LAT_MIN),
            'longitude': self.LON_MIN + 0.3 * (self.LON_MAX - self.LON_MIN),
            'baro_altitude': 5000 + np.random.randint(0, 3000),
            'velocity': 200 + np.random.randint(0, 100),
            'true_track': np.random.randint(0, 360),
            'vertical_rate': 0
        }
    
    def _generate_default_destination(self):
        return {
            'latitude': self.LAT_MAX - 0.2 * (self.LAT_MAX - self.LAT_MIN),
            'longitude': self.LON_MAX - 0.3 * (self.LON_MAX - self.LON_MIN)
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize aircraft state
        self.current_position = self.start_point.copy()
        
        # Reset tracking variables
        self.step_count = 0
        self.total_turbulence = 0
        self.total_fuel_used = 0
        self.trajectory = [self.current_position.copy()]
        
        # Update initial weather
        self._update_weather_at_position()
        
        # Get initial observation
        return self._get_observation(), {}
    
    def step(self, action):
        # Decode action into heading and altitude changes
        heading_idx = action // len(self.altitude_changes)
        altitude_idx = action % len(self.altitude_changes)
        
        heading_change = self.heading_changes[heading_idx]
        altitude_change = self.altitude_changes[altitude_idx]
        
        # Store previous state for reward calculation
        prev_position = self.current_position.copy()
        prev_distance = self._calculate_distance_to_destination()
        
        # Update aircraft state
        self._update_aircraft_state(heading_change, altitude_change)
        
        # Increment step count
        self.step_count += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_position, prev_distance, heading_change, altitude_change
        )
        
        # Check episode termination conditions
        current_distance = self._calculate_distance_to_destination()
        reached_destination = current_distance < 2.0  # 2 km threshold
        out_of_bounds = not self._check_bounds()
        max_steps_reached = self.step_count >= self.max_steps
        
        # Determine done and truncated states
        terminated = reached_destination or out_of_bounds
        truncated = max_steps_reached
        
        # Store trajectory point
        current_pos_with_turbulence = self.current_position.copy()
        current_pos_with_turbulence['turbulence'] = self.current_turbulence
        self.trajectory.append(current_pos_with_turbulence)
        
        # Prepare info dictionary
        info = {
            'current_position': self.current_position,
            'distance_to_destination': current_distance,
            'turbulence': self.current_turbulence,
            'fuel_used': self.current_fuel_rate,
            'total_fuel': self.total_fuel_used,
            'total_turbulence': self.total_turbulence,
            'out_of_bounds': out_of_bounds,
            'reached_destination': reached_destination
        }
        
        return observation, reward, terminated, truncated, info
    
    def _update_aircraft_state(self, heading_change, altitude_change):
        # Update heading
        current_heading = self.current_position['true_track']
        new_heading = (current_heading + heading_change) % 360
        # Update altitude
        current_altitude = self.current_position['baro_altitude']
        new_altitude = max(500, min(40000, current_altitude + altitude_change))
        # Calculate vertical rate
        vertical_rate = altitude_change / 5  # feet per minute
        # Calculate new position based on heading and velocity
        velocity_ms = self.current_position['velocity']  
        # Adjust time step to a smaller value for finer movement per step
        distance_km = velocity_ms * self.time_step / 1000 # Convert m/s to km
        # Convert heading to radians
        heading_rad = radians(new_heading)
        # Earth's radius in km
        R = 6371.0
        # Calculate new latitude and longitude
        current_lat = self.current_position['latitude']
        current_lon = self.current_position['longitude']
        new_lat = current_lat + (distance_km / R) * cos(heading_rad) * (180 / np.pi)
        new_lon = current_lon + (distance_km / R) * sin(heading_rad) * (180 / np.pi) / cos(radians(current_lat))
        # Update aircraft position
        self.current_position.update({
            'latitude': new_lat,
            'longitude': new_lon,
            'baro_altitude': new_altitude,
            'true_track': new_heading,
            'vertical_rate': vertical_rate
        })
        # Update weather at new position
        self._update_weather_at_position()
        # Calculate fuel consumption
        self._calculate_fuel_rate(heading_change, altitude_change)
        # Predict turbulence at new position
        self.current_turbulence = self._predict_turbulence()
        self.total_turbulence += self.current_turbulence
    
    def _update_weather_at_position(self):
        lat = self.current_position['latitude']
        lon = self.current_position['longitude']
        
        # Find closest weather grid point
        min_dist = float('inf')
        closest_weather = None
        
        for weather_point in self.weather_grid:
            dist = self._haversine_distance(
                lat, lon, 
                weather_point['lat'], weather_point['lon']
            )
            if dist < min_dist:
                min_dist = dist
                closest_weather = weather_point
        
        # Update position with weather data
        self.current_position.update({
            'temperature': closest_weather['temperature'],
            'pressure': closest_weather['pressure'],
            'humidity': closest_weather['humidity'],
            'wind_speed': closest_weather['wind_speed'],
            'wind_deg': closest_weather['wind_deg']
        })
        
        # Store current weather context
        self.current_weather = closest_weather
    
    def _predict_turbulence(self):
        # Prepare feature vector in correct order
        feature_dict = {
            'velocity': self.current_position['velocity'],
            'baro_altitude': self.current_position['baro_altitude'],
            'vertical_rate': self.current_position['vertical_rate'],
            'temperature': self.current_position['temperature'],
            'pressure': self.current_position['pressure'],
            'humidity': self.current_position['humidity'],
            'wind_speed': self.current_position['wind_speed'],
            'wind_deg': self.current_position['wind_deg']
        }
        
        # Create a NumPy array in the correct order
        features = np.array([feature_dict[name] for name in self.feature_names]).reshape(1, -1)
        
        # Convert to DataFrame with the same column names the scaler was fitted on
        features_df = pd.DataFrame(features, columns=self.feature_names)
        
        # Now transform using the scaler
        scaled_features = self.scaler.transform(features_df)

        # Predict turbulence using the scaled features
        turbulence = self.turbulence_model.predict(scaled_features)[0]
        
        return max(0, min(turbulence, 1))
        
    def _predict_turbulence_ahead(self):
        heading = self.current_position['true_track']
        lat = self.current_position['latitude']
        lon = self.current_position['longitude']
        
        # Points at 1km, 2km, and 5km ahead
        distances = [1, 2, 5]  # km
        turbulence_ahead = []
        
        for distance in distances:
            # Calculate position at distance ahead
            R = 6371.0  # Earth's radius in km
            heading_rad = radians(heading)
            
            new_lat = lat + (distance / R) * cos(heading_rad) * (180 / np.pi)
            new_lon = lon + (distance / R) * sin(heading_rad) * (180 / np.pi) / cos(radians(lat))
            
            # Find closest weather point
            min_dist = float('inf')
            closest_weather = None
            
            for weather_point in self.weather_grid:
                dist = self._haversine_distance(new_lat, new_lon, weather_point['lat'], weather_point['lon'])
                if dist < min_dist:
                    min_dist = dist
                    closest_weather = weather_point
            
            # Prepare feature vector
            feature_dict = {
                'velocity': self.current_position['velocity'],
                'baro_altitude': self.current_position['baro_altitude'],
                'vertical_rate': self.current_position['vertical_rate'],
                'temperature': closest_weather['temperature'],
                'pressure': closest_weather['pressure'],
                'humidity': closest_weather['humidity'],
                'wind_speed': closest_weather['wind_speed'],
                'wind_deg': closest_weather['wind_deg']
            }
            
            # Ensure correct feature order
            features = np.array([feature_dict[name] for name in self.feature_names]).reshape(1, -1)
            features_df = pd.DataFrame(features, columns=self.feature_names)

            # Scale features
            scaled_features = self.scaler.transform(features_df)

            # Predict turbulence
            turbulence = self.turbulence_model.predict(scaled_features)[0]
            turbulence_ahead.append(max(0, min(turbulence, 1)))
        
        return np.array(turbulence_ahead)
    
    def _get_observation(self):
        # Current aircraft state
        aircraft_state = np.array([
            self.current_position['latitude'],
            self.current_position['longitude'],
            self.current_position['baro_altitude'],
            self.current_position['velocity'],
            self.current_position['true_track'],
            self.current_position['vertical_rate']
        ])
        
        # Current weather state
        weather_state = np.array([
            self.current_position['temperature'],
            self.current_position['pressure'],
            self.current_position['humidity'],
            self.current_position['wind_speed'],
            self.current_position['wind_deg']
        ])
        
        # Distance and bearing to destination
        distance = self._calculate_distance_to_destination()
        bearing = self._calculate_bearing_to_destination()
        destination_info = np.array([distance, bearing])
        
        # Predicted turbulence ahead
        turbulence_ahead = self._predict_turbulence_ahead()
        
        # Combine all observation components
        observation = np.concatenate([
            aircraft_state, 
            weather_state, 
            destination_info, 
            turbulence_ahead
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_fuel_rate(self, heading_change, altitude_change):
 
        # Base fuel rate depends on velocity and altitude
        velocity = self.current_position['velocity']
        altitude = self.current_position['baro_altitude']
        
        # Base consumption model
        base_rate = 0.01 * velocity + 0.00001 * velocity**2 - 0.00005 * altitude
        base_rate = max(0.5, base_rate)  # Ensure minimum consumption
        
        # Maneuver penalty
        maneuver_factor = (
            0.02 * abs(heading_change) +  # Heading change penalty
            0.0005 * abs(altitude_change)  # Altitude change penalty
        )
        
        # Total fuel rate
        self.current_fuel_rate = base_rate * (1 + maneuver_factor)
        self.total_fuel_used += self.current_fuel_rate
        
        return self.current_fuel_rate
    
    def _calculate_reward(self, prev_position, prev_distance, heading_change, altitude_change):
        # Current distance to destination
        current_distance = self._calculate_distance_to_destination()
        
        distance_change = prev_distance - current_distance
        distance_reward = self.reward_weights['distance'] * 3.0 * distance_change
        
        target_bearing = self._calculate_bearing_to_destination()
        current_heading = self.current_position['true_track']
        heading_diff = min(abs(target_bearing - current_heading), 
                        360 - abs(target_bearing - current_heading))
        heading_alignment = max(0, 1 - heading_diff / 180)  
        direction_reward = 2.0 * heading_alignment  

        turbulence_penalty = -self.reward_weights['turbulence'] * 0.7 * self.current_turbulence
        fuel_penalty = -self.reward_weights['fuel'] * 0.6 * self.current_fuel_rate

        proximity_factor = max(0, 1 - (current_distance / 20))  
        proximity_bonus = 5.0 * proximity_factor**2  

        completion_bonus = self.reward_weights.get('completion', 150.0) if current_distance < 2.0 else 0

        bounds_penalty = -100 if not self._check_bounds() else 0
        
        # Combine all reward components
        total_reward = (
            distance_reward +
            direction_reward +
            turbulence_penalty +
            fuel_penalty +
            # alive_bonus +  # Removed
            proximity_bonus +
            completion_bonus +
            bounds_penalty
        )
    
        return total_reward

    def _calculate_distance_to_destination(self):
        return self._haversine_distance(
            self.current_position['latitude'], 
            self.current_position['longitude'],
            self.destination['latitude'], 
            self.destination['longitude']
        )
    
    def _calculate_bearing_to_destination(self):
        lat1 = radians(self.current_position['latitude'])
        lon1 = radians(self.current_position['longitude'])
        lat2 = radians(self.destination['latitude'])
        lon2 = radians(self.destination['longitude'])
        
        y = sin(lon2 - lon1) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
        
        bearing = degrees(atan2(y, x))
        return (bearing + 360) % 360
    
    def _check_bounds(self):
        lat = self.current_position['latitude']
        lon = self.current_position['longitude']
        
        return (self.LAT_MIN <= lat <= self.LAT_MAX and
                self.LON_MIN <= lon <= self.LON_MAX)
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth's radius in kilometers
        R = 6371.0
        
        return R * c
    
    def save_trajectory(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
        
        trajectory_data = {
            'start': self.start_point,
            'destination': self.destination,
            'points': self.trajectory,
            'total_turbulence': self.total_turbulence,
            'total_fuel_used': self.total_fuel_used,
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'reached_destination': self._calculate_distance_to_destination() < 2.0
        }
        
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        return filename
    
    def render(self, mode='human'):

        if mode != 'human':
            return
        
        # Print current state details
        print("\n=== Flight State ===")
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"Position: ({self.current_position['latitude']:.4f}, {self.current_position['longitude']:.4f})")
        print(f"Altitude: {self.current_position['baro_altitude']:.0f} ft")
        print(f"Heading: {self.current_position['true_track']:.1f}°")
        print(f"Speed: {self.current_position['velocity']:.1f} m/s")
        print(f"Vertical Rate: {self.current_position['vertical_rate']:.1f} ft/min")
        
        # Weather information
        print("\n=== Weather ===")
        print(f"Temperature: {self.current_position['temperature']:.1f}°C")
        print(f"Pressure: {self.current_position['pressure']:.1f} hPa")
        print(f"Humidity: {self.current_position['humidity']:.1f}%")
        print(f"Wind: {self.current_position['wind_speed']:.1f} m/s @ {self.current_position['wind_deg']:.1f}°")
        
        # Flight performance
        print("\n=== Performance ===")
        print(f"Turbulence: {self.current_turbulence:.3f}")
        print(f"Fuel Used: {self.total_fuel_used:.2f}")
        print(f"Distance to Destination: {self._calculate_distance_to_destination():.2f} km")
    
    def close(self):
        """
        Clean up resources when environment is closed.
        """
        pass
    
    def get_normalized_trajectory(self):

        lat_range = self.LAT_MAX - self.LAT_MIN
        lon_range = self.LON_MAX - self.LON_MIN
        
        normalized_points = []
        for point in self.trajectory:
            norm_point = {
                'x': (point['longitude'] - self.LON_MIN) / lon_range,
                'y': (point['latitude'] - self.LAT_MIN) / lat_range,
                'altitude': point.get('baro_altitude', 0) / 40000,  # Normalize altitude
                'turbulence': point.get('turbulence', 0)  # Already 0-1
            }
            normalized_points.append(norm_point)
        
        return {
            'points': normalized_points,
            'start': {
                'x': (self.start_point['longitude'] - self.LON_MIN) / lon_range,
                'y': (self.start_point['latitude'] - self.LAT_MIN) / lat_range
            },
            'destination': {
                'x': (self.destination['longitude'] - self.LON_MIN) / lon_range,
                'y': (self.destination['latitude'] - self.LAT_MIN) / lat_range
            }
        }
    
    def generate_flight_summary(self):

        return {
            'start_point': self.start_point,
            'destination': self.destination,
            'total_steps': self.step_count,
            'total_turbulence': self.total_turbulence,
            'total_fuel_used': self.total_fuel_used,
            'reached_destination': self._calculate_distance_to_destination() < 2.0,
            'final_position': self.current_position,
            'final_distance_to_destination': self._calculate_distance_to_destination(),
            'trajectory_length': len(self.trajectory)
        }
    
    def example_usage(self):

        print("Running example flight route environment...")
        
        # Reset the environment
        obs, _ = self.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Choose a random action
            action = self.action_space.sample()
            
            # Take a step in the environment
            obs, reward, done, truncated, info = self.step(action)
            
            # Accumulate reward
            total_reward += reward
            
            # Render current state
            self.render()
            
            # Print action and reward details
            print(f"Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print("---")
        
        # Save trajectory
        trajectory_file = self.save_trajectory("example_trajectory.json")
        print(f"Trajectory saved to {trajectory_file}")
        
        # Generate and print summary
        summary = self.generate_flight_summary()
        print("\n=== Flight Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return total_reward

# Utility function for creating example environment
def create_example_environment(
    turbulence_model_path='models/turbulence_model.joblib',
    scaler_path='models/turbulence_scaler.joblib'
):

    # Create synthetic weather grid
    weather_grid = create_synthetic_weather_grid()
    
    # Define start and destination points
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
        env = FlightRouteEnv(
            turbulence_model_path=turbulence_model_path,
            scaler_path=scaler_path,
            weather_grid=weather_grid,
            start_point=start_point,
            destination=destination
        )
        return env
    except FileNotFoundError:
        print("Turbulence model not found. Please train the model first.")
        print("You can use: python turbulence_model.py to train the model.")
        return None

# Expose key functions and classes
__all__ = [
    'FlightRouteEnv', 
    'create_synthetic_weather_grid', 
    'create_example_environment'
]

# Run as a script for testing
if __name__ == "__main__":
    # Create and run example environment
    env = create_example_environment()
    
    if env is not None:
        # Run example usage
        total_reward = env.example_usage()
        print(f"\nTotal Episode Reward: {total_reward:.2f}")
