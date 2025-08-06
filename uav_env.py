# uav_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- ĐỊNH NGHĨA CÁC HẰNG SỐ ---
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

NUM_NO_FLY_ZONES = 3
MAX_ZONE_RADIUS = 100

# --- THAY ĐỔI: Thêm Trọng số mới cho Reward Shaping ---
W_THROUGHPUT = 1.0          # Thưởng cho dữ liệu thu thập được
W_UAV_ENERGY = 0.01         # Phạt cho năng lượng UAV tiêu thụ
W_OUT_OF_BOUNDS = 100       # Phạt nặng khi bay ra khỏi bản đồ
W_IN_NO_FLY_ZONE = 200      # Phạt nặng hơn khi bay vào vùng cấm
W_CLOSER_TO_IOT = 0.1       # Thưởng khi tiến lại gần IoT
W_AWAY_FROM_NFZ = 0.05      # Thưởng khi đi ra xa vùng cấm (hoặc phạt khi lại gần)

class UAVNetworkEnv(gym.Env):
    """
    Môi trường mô phỏng Mạng UAV-IoT.
    v5: Áp dụng Reward Shaping để dẫn đường cho agent.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_uavs=2, num_iot_nodes=15):
        super(UAVNetworkEnv, self).__init__()
        # Các phần khởi tạo khác giữ nguyên từ v4
        self.num_uavs = num_uavs
        self.num_iot_nodes = num_iot_nodes
        self.no_fly_zones = []
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_uavs * 2,), dtype=np.float32)
        obs_dim = (self.num_uavs * 3) + (self.num_iot_nodes * 4) + (NUM_NO_FLY_ZONES * 3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
    def _get_obs(self):
        # Giữ nguyên từ v4
        uav_states = np.array([[u['x']/WORLD_SIZE_X, u['y']/WORLD_SIZE_Y, u['energy']/UAV_ENERGY_START] for u in self.uavs], dtype=np.float32).flatten()
        iot_states = np.array([[i['x']/WORLD_SIZE_X, i['y']/WORLD_SIZE_Y, i['battery']/IOT_BATTERY_START, i['data']/IOT_DATA_START] for i in self.iot_nodes], dtype=np.float32).flatten()
        zone_states = np.array([[z['center'][0]/WORLD_SIZE_X, z['center'][1]/WORLD_SIZE_Y, z['radius']/MAX_ZONE_RADIUS] for z in self.no_fly_zones], dtype=np.float32).flatten()
        return np.concatenate([uav_states, iot_states, zone_states])

    def _get_info(self):
        # Giữ nguyên từ v4
        return {
            "uav_positions": [(u['x'], u['y']) for u in self.uavs],
            "iot_data_left": sum(i['data'] for i in self.iot_nodes),
            "total_energy_left": sum(u['energy'] for u in self.uavs),
            "no_fly_zones": self.no_fly_zones 
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.uavs = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'energy': UAV_ENERGY_START} for _ in range(self.num_uavs)]
        self.iot_nodes = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'battery': IOT_BATTERY_START, 'data': IOT_DATA_START} for _ in range(self.num_iot_nodes)]
        self.no_fly_zones = []
        for _ in range(NUM_NO_FLY_ZONES):
            center_x = self.np_random.uniform(100, WORLD_SIZE_X - 100)
            center_y = self.np_random.uniform(100, WORLD_SIZE_Y - 100)
            radius = self.np_random.uniform(50, MAX_ZONE_RADIUS)
            self.no_fly_zones.append({'center': (center_x, center_y), 'radius': radius})

        # --- THAY ĐỔI: Lưu lại trạng thái khoảng cách ban đầu ---
        self._update_distances()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _update_distances(self):
        """Hàm pomocniczy để tính và lưu lại khoảng cách."""
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            # Khoảng cách đến IoT gần nhất còn dữ liệu
            min_dist_iot = float('inf')
            for iot in self.iot_nodes:
                if iot['data'] > 0:
                    dist = np.sqrt((uav['x'] - iot['x'])**2 + (uav['y'] - iot['y'])**2)
                    if dist < min_dist_iot:
                        min_dist_iot = dist
            uav['prev_dist_to_iot'] = min_dist_iot

            # Khoảng cách đến vùng cấm gần nhất
            min_dist_nfz = float('inf')
            for zone in self.no_fly_zones:
                dist = np.sqrt((uav['x'] - zone['center'][0])**2 + (uav['y'] - zone['center'][1])**2) - zone['radius']
                if dist < min_dist_nfz:
                    min_dist_nfz = dist
            uav['prev_dist_to_nfz'] = min_dist_nfz


    def step(self, action):
        self.current_step += 1
        
        # ======================= HÀM REWARD MỚI =======================
        # Bắt đầu với reward cơ bản (sự kiện lớn)
        reward = 0
        total_throughput = 0
        total_energy_consumed = 0

        # --- REWARD SHAPING ---
        # Tính toán reward dựa trên sự thay đổi khoảng cách
        shaping_reward = 0
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            
            # 1. Tính khoảng cách mới
            # Khoảng cách đến IoT gần nhất còn dữ liệu
            new_min_dist_iot = float('inf')
            for iot in self.iot_nodes:
                if iot['data'] > 0:
                    dist = np.sqrt((uav['x'] - iot['x'])**2 + (uav['y'] - iot['y'])**2)
                    if dist < new_min_dist_iot:
                        new_min_dist_iot = dist
            
            # Khoảng cách đến vùng cấm gần nhất
            new_min_dist_nfz = float('inf')
            for zone in self.no_fly_zones:
                dist = np.sqrt((uav['x'] - zone['center'][0])**2 + (uav['y'] - zone['center'][1])**2) - zone['radius']
                if dist < new_min_dist_nfz:
                    new_min_dist_nfz = dist

            # 2. Tính reward từ việc thay đổi khoảng cách
            # Thưởng nếu tiến lại gần IoT
            if 'prev_dist_to_iot' in uav and new_min_dist_iot < uav['prev_dist_to_iot']:
                 shaping_reward += (uav['prev_dist_to_iot'] - new_min_dist_iot) * W_CLOSER_TO_IOT
            
            # Phạt nếu tiến lại gần vùng cấm
            if 'prev_dist_to_nfz' in uav and new_min_dist_nfz < uav['prev_dist_to_nfz']:
                 shaping_reward -= (uav['prev_dist_to_nfz'] - new_min_dist_nfz) * W_AWAY_FROM_NFZ
        
        # Lưu lại khoảng cách mới cho bước tiếp theo
        self._update_distances()
        
        # Di chuyển UAV và tính các hình phạt lớn
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            delta_x, delta_y = action[i*2] * UAV_SPEED, action[i*2 + 1] * UAV_SPEED
            uav['x'] += delta_x
            uav['y'] += delta_y
            
            energy_consumed_this_step = UAV_ENERGY_CONSUMPTION_HOVER if delta_x == 0 and delta_y == 0 else UAV_ENERGY_CONSUMPTION_FLY
            uav['energy'] -= energy_consumed_this_step
            total_energy_consumed += energy_consumed_this_step
            
            if not (0 <= uav['x'] <= WORLD_SIZE_X and 0 <= uav['y'] <= WORLD_SIZE_Y):
                reward -= W_OUT_OF_BOUNDS
                uav['x'] = np.clip(uav['x'], 0, WORLD_SIZE_X)
                uav['y'] = np.clip(uav['y'], 0, WORLD_SIZE_Y)

            for zone in self.no_fly_zones:
                if np.sqrt((uav['x'] - zone['center'][0])**2 + (uav['y'] - zone['center'][1])**2) < zone['radius']:
                    reward -= W_IN_NO_FLY_ZONE

        # Mô phỏng Giao tiếp (Thưởng lớn)
        for iot_node in self.iot_nodes:
            if iot_node['data'] > 0 and iot_node['battery'] > 0:
                for uav in self.uavs:
                    if np.sqrt((uav['x'] - iot_node['x'])**2 + (uav['y'] - iot_node['y'])**2) < COMM_RANGE:
                        data_transmitted = min(iot_node['data'], IOT_DATA_RATE)
                        iot_node['data'] -= data_transmitted
                        iot_node['battery'] -= IOT_ENERGY_CONSUMPTION_TX
                        total_throughput += data_transmitted
                        break

        # TỔNG HỢP REWARD CUỐI CÙNG
        reward += total_throughput * W_THROUGHPUT
        reward -= total_energy_consumed * W_UAV_ENERGY
        reward += shaping_reward # Cộng thêm phần thưởng "dẫn đường"

        # Điều kiện kết thúc
        terminated = all(uav['energy'] <= 0 for uav in self.uavs) or all(iot['data'] <= 0 for iot in self.iot_nodes)
        truncated = self.current_step >= MAX_STEPS

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
