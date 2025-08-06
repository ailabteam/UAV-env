# uav_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Bỏ 'import random' vì chúng ta sẽ dùng bộ sinh số ngẫu nhiên của Gymnasium

# --- ĐỊNH NGHĨA CÁC HẰNG SỐ (giữ nguyên) ---
WORLD_SIZE_X = 1000
WORLD_SIZE_Y = 1000
MAX_STEPS = 500

UAV_SPEED = 10
UAV_ENERGY_START = 1000.0
UAV_ENERGY_CONSUMPTION_FLY = 1.0
UAV_ENERGY_CONSUMPTION_HOVER = 0.5

IOT_DATA_START = 100.0
IOT_BATTERY_START = 100.0
IOT_DATA_RATE = 10
IOT_ENERGY_CONSUMPTION_TX = 0.1
COMM_RANGE = 250

W_THROUGHPUT = 1.0
W_UAV_ENERGY = 0.01
W_OUT_OF_BOUNDS = 100

class UAVNetworkEnv(gym.Env):
    """
    Môi trường mô phỏng Mạng UAV-IoT tùy chỉnh theo API của Gymnasium.
    v3: Sửa lỗi non-deterministic bằng cách sử dụng self.np_random.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_uavs=1, num_iot_nodes=10):
        super(UAVNetworkEnv, self).__init__()

        self.num_uavs = num_uavs
        self.num_iot_nodes = num_iot_nodes

        # Action Space (giữ nguyên)
        self.action_space = spaces.Box(low=-1, high=1, 
                                       shape=(self.num_uavs * 2,), 
                                       dtype=np.float32)

        # Observation Space (giữ nguyên)
        obs_dim = (self.num_uavs * 3) + (self.num_iot_nodes * 4)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(obs_dim,), 
                                            dtype=np.float32)
        
    def _get_obs(self):
        # Giữ nguyên từ v2
        uav_states = np.array([
            [uav['x'] / WORLD_SIZE_X, uav['y'] / WORLD_SIZE_Y, uav['energy'] / UAV_ENERGY_START] 
            for uav in self.uavs
        ], dtype=np.float32).flatten()
        iot_states = np.array([
            [iot['x'] / WORLD_SIZE_X, iot['y'] / WORLD_SIZE_Y, iot['battery'] / IOT_BATTERY_START, iot['data'] / IOT_DATA_START] 
            for iot in self.iot_nodes
        ], dtype=np.float32).flatten()
        return np.concatenate([uav_states, iot_states])

    def _get_info(self):
        # Giữ nguyên từ v2
        return {
            "uav_positions": [ (uav['x'], uav['y']) for uav in self.uavs ],
            "iot_data_left": sum(iot['data'] for iot in self.iot_nodes),
            "total_energy_left": sum(uav['energy'] for uav in self.uavs)
        }

    def reset(self, seed=None, options=None):
        """Reset môi trường về trạng thái ban đầu."""
        # ======================= SỬA ĐỔI CHÍNH Ở ĐÂY =======================
        # Thiết lập seed cho bộ sinh số ngẫu nhiên của môi trường
        super().reset(seed=seed)
        self.current_step = 0

        # Khởi tạo UAV bằng self.np_random thay vì 'random'
        self.uavs = [{
            'x': self.np_random.uniform(0, WORLD_SIZE_X),
            'y': self.np_random.uniform(0, WORLD_SIZE_Y),
            'energy': UAV_ENERGY_START
        } for _ in range(self.num_uavs)]

        # Khởi tạo các node IoT bằng self.np_random thay vì 'random'
        self.iot_nodes = [{
            'x': self.np_random.uniform(0, WORLD_SIZE_X),
            'y': self.np_random.uniform(0, WORLD_SIZE_Y),
            'battery': IOT_BATTERY_START,
            'data': IOT_DATA_START
        } for _ in range(self.num_iot_nodes)]
        # =================================================================

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Hàm step không có yếu tố ngẫu nhiên nên không cần thay đổi
        self.current_step += 1
        reward = 0
        total_throughput = 0

        total_energy_consumed = 0
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            delta_x = action[i*2] * UAV_SPEED
            delta_y = action[i*2 + 1] * UAV_SPEED
            uav['x'] += delta_x
            uav['y'] += delta_y
            energy_consumed_this_step = UAV_ENERGY_CONSUMPTION_HOVER if delta_x == 0 and delta_y == 0 else UAV_ENERGY_CONSUMPTION_FLY
            uav['energy'] -= energy_consumed_this_step
            total_energy_consumed += energy_consumed_this_step
            if not (0 <= uav['x'] <= WORLD_SIZE_X and 0 <= uav['y'] <= WORLD_SIZE_Y):
                reward -= W_OUT_OF_BOUNDS
                uav['x'] = np.clip(uav['x'], 0, WORLD_SIZE_X)
                uav['y'] = np.clip(uav['y'], 0, WORLD_SIZE_Y)

        for iot_node in self.iot_nodes:
            if iot_node['data'] <= 0 or iot_node['battery'] <= 0:
                continue
            min_dist = float('inf')
            best_uav = None
            for uav in self.uavs:
                dist = np.sqrt((uav['x'] - iot_node['x'])**2 + (uav['y'] - iot_node['y'])**2)
                if dist < COMM_RANGE and dist < min_dist:
                    min_dist = dist
                    best_uav = uav
            if best_uav:
                data_transmitted = min(iot_node['data'], IOT_DATA_RATE)
                iot_node['data'] -= data_transmitted
                iot_node['battery'] -= IOT_ENERGY_CONSUMPTION_TX
                total_throughput += data_transmitted

        reward += total_throughput * W_THROUGHPUT
        reward -= total_energy_consumed * W_UAV_ENERGY
        
        terminated = all(uav['energy'] <= 0 for uav in self.uavs) or all(iot['data'] <= 0 for iot in self.iot_nodes)
        truncated = self.current_step >= MAX_STEPS

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
