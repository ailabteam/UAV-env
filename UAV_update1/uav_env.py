# uav_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

# --- HẰNG SỐ ---
WORLD_SIZE_X = 1000; WORLD_SIZE_Y = 1000; MAX_STEPS = 500
UAV_SPEED = 10; UAV_ENERGY_START = 1000.0; UAV_ENERGY_CONSUMPTION_FLY = 1.0; UAV_ENERGY_CONSUMPTION_HOVER = 0.5
IOT_DATA_START = 100.0; IOT_BATTERY_START = 100.0; IOT_DATA_RATE = 10; COMM_RANGE = 250
NUM_NO_FLY_ZONES = 3; MAX_ZONE_RADIUS = 100
W_THROUGHPUT = 1.0; W_UAV_ENERGY = 0.01; W_OUT_OF_BOUNDS = 100; W_IN_NO_FLY_ZONE = 200
N_CLOSEST_IOT = 5; N_CLOSEST_NFZ = 3

class UAVNetworkEnv(gym.Env):
    """
    Phiên bản Môi trường v7: Tĩnh, có chướng ngại vật, và Observation được thiết kế thông minh.
    Tích hợp force_state để đảm bảo đánh giá đồng bộ.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_uavs=2, num_iot_nodes=15):
        super(UAVNetworkEnv, self).__init__()
        self.num_uavs = num_uavs; self.num_iot_nodes = num_iot_nodes
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_uavs * 2,), dtype=np.float32)
        single_uav_obs_dim = 3 + (N_CLOSEST_IOT * 4) + (N_CLOSEST_NFZ * 3)
        obs_dim = self.num_uavs * single_uav_obs_dim
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        all_obs = []
        for uav in self.uavs:
            uav_state = [uav['x']/WORLD_SIZE_X*2-1, uav['y']/WORLD_SIZE_Y*2-1, uav['energy']/UAV_ENERGY_START*2-1]
            self.iot_nodes.sort(key=lambda i: np.hypot(i['x'] - uav['x'], i['y'] - uav['y']))
            iot_states = [];
            for i in range(N_CLOSEST_IOT):
                if i < len(self.iot_nodes): iot = self.iot_nodes[i]; iot_states.extend([(iot['x']-uav['x'])/WORLD_SIZE_X, (iot['y']-uav['y'])/WORLD_SIZE_Y, iot['battery']/IOT_BATTERY_START, iot['data']/IOT_DATA_START])
                else: iot_states.extend([0,0,0,0])
            self.no_fly_zones.sort(key=lambda z: np.hypot(z['center'][0] - uav['x'], z['center'][1] - uav['y']))
            nfz_states = []
            for i in range(N_CLOSEST_NFZ):
                if i < len(self.no_fly_zones): zone = self.no_fly_zones[i]; nfz_states.extend([(zone['center'][0]-uav['x'])/WORLD_SIZE_X, (zone['center'][1]-uav['y'])/WORLD_SIZE_Y, zone['radius']/MAX_ZONE_RADIUS])
                else: nfz_states.extend([0,0,0])
            all_obs.append(np.concatenate([uav_state, iot_states, nfz_states]))
        return np.concatenate(all_obs).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.current_step = 0
        self.uavs = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'energy': UAV_ENERGY_START} for _ in range(self.num_uavs)]
        self.iot_nodes = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'battery': IOT_BATTERY_START, 'data': IOT_DATA_START} for _ in range(self.num_iot_nodes)]
        self.no_fly_zones = [{'center': (self.np_random.uniform(100, WORLD_SIZE_X-100), self.np_random.uniform(100, WORLD_SIZE_Y-100)), 'radius': self.np_random.uniform(50, MAX_ZONE_RADIUS)} for _ in range(NUM_NO_FLY_ZONES)]
        return self._get_obs(), self._get_info()

    def force_state(self, uavs, iot_nodes, no_fly_zones):
        self.current_step = 0
        self.uavs = copy.deepcopy(uavs); self.iot_nodes = copy.deepcopy(iot_nodes); self.no_fly_zones = copy.deepcopy(no_fly_zones)
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1; reward = 0; total_throughput = 0; total_energy_consumed = 0
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            delta_x, delta_y = action[i*2] * UAV_SPEED, action[i*2 + 1] * UAV_SPEED
            uav['x'] += delta_x; uav['y'] += delta_y
            energy_consumed = UAV_ENERGY_CONSUMPTION_HOVER if delta_x == 0 and delta_y == 0 else UAV_ENERGY_CONSUMPTION_FLY
            uav['energy'] -= energy_consumed; total_energy_consumed += energy_consumed
            if not (0 <= uav['x'] <= WORLD_SIZE_X and 0 <= uav['y'] <= WORLD_SIZE_Y):
                reward -= W_OUT_OF_BOUNDS; uav['x'] = np.clip(uav['x'], 0, WORLD_SIZE_X); uav['y'] = np.clip(uav['y'], 0, WORLD_SIZE_Y)
            for zone in self.no_fly_zones:
                if np.hypot(uav['x'] - zone['center'][0], uav['y'] - zone['center'][1]) < zone['radius']:
                    reward -= W_IN_NO_FLY_ZONE
        for iot_node in self.iot_nodes:
            if iot_node['data'] > 0:
                for uav in self.uavs:
                    if np.hypot(uav['x'] - iot_node['x'], uav['y'] - iot_node['y']) < COMM_RANGE:
                        data_tx = min(iot_node['data'], IOT_DATA_START); iot_node['data'] -= data_tx
                        total_throughput += data_tx; break
        reward += total_throughput * W_THROUGHPUT
        reward -= total_energy_consumed * W_UAV_ENERGY
        terminated = all(u['energy'] <= 0 for u in self.uavs) or all(i['data'] <= 0 for i in self.iot_nodes)
        truncated = self.current_step >= MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_info(self):
        return { "uav_positions": [(u['x'], u['y']) for u in self.uavs], "iot_data_left": sum(i['data'] for i in self.iot_nodes) }
    def close(self): pass
