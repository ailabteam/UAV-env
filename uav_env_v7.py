# uav_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- HẰNG SỐ ---
# ... (Giữ nguyên các hằng số từ v6) ...
WORLD_SIZE_X = 1000; WORLD_SIZE_Y = 1000; MAX_STEPS = 500
UAV_SPEED = 10; UAV_ENERGY_START = 1000.0; UAV_ENERGY_CONSUMPTION_FLY = 1.0; UAV_ENERGY_CONSUMPTION_HOVER = 0.5
IOT_DATA_START = 100.0; IOT_BATTERY_START = 100.0; IOT_DATA_RATE = 10; IOT_ENERGY_CONSUMPTION_TX = 0.1; COMM_RANGE = 250
NUM_NO_FLY_ZONES = 3; MAX_ZONE_RADIUS = 100
W_THROUGHPUT = 1.0; W_UAV_ENERGY = 0.01; W_OUT_OF_BOUNDS = 100; W_IN_NO_FLY_ZONE = 200

# --- THAY ĐỔI: Hằng số cho Feature Engineering ---
N_CLOSEST_IOT = 5  # Agent sẽ "nhìn thấy" 5 node IoT gần nhất
N_CLOSEST_NFZ = 3  # Agent sẽ "nhìn thấy" 3 vùng cấm gần nhất

class UAVNetworkEnv(gym.Env):
    """
    v7: Sử dụng Feature Engineering để thiết kế lại Observation Space.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_uavs=2, num_iot_nodes=15):
        super(UAVNetworkEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_iot_nodes = num_iot_nodes
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_uavs * 2,), dtype=np.float32)

        # --- THAY ĐỔI LỚN: Định nghĩa Observation Space mới ---
        # Mỗi UAV sẽ có một "khối" quan sát riêng, sau đó tất cả được nối lại.
        # Khối của 1 UAV bao gồm:
        # 1. Trạng thái bản thân: [x, y, energy] (3)
        # 2. Trạng thái của K IoT gần nhất: K * [dx, dy, battery, data] (5 * 4 = 20)
        # 3. Trạng thái của M Vùng cấm gần nhất: M * [dx, dy, radius] (3 * 3 = 9)
        single_uav_obs_dim = 3 + (N_CLOSEST_IOT * 4) + (N_CLOSEST_NFZ * 3)
        obs_dim = self.num_uavs * single_uav_obs_dim
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        all_obs = []
        # Tạo khối quan sát cho từng UAV
        for uav in self.uavs:
            # 1. Trạng thái bản thân (chuẩn hóa)
            uav_state = [
                uav['x'] / WORLD_SIZE_X * 2 - 1, # Chuẩn hóa về [-1, 1]
                uav['y'] / WORLD_SIZE_Y * 2 - 1,
                uav['energy'] / UAV_ENERGY_START * 2 - 1,
            ]
            
            # 2. Tìm K IoT gần nhất
            self.iot_nodes.sort(key=lambda i: np.hypot(i['x'] - uav['x'], i['y'] - uav['y']))
            closest_iot_states = []
            for i in range(N_CLOSEST_IOT):
                if i < len(self.iot_nodes):
                    iot = self.iot_nodes[i]
                    closest_iot_states.extend([
                        (iot['x'] - uav['x']) / WORLD_SIZE_X, # Vị trí tương đối
                        (iot['y'] - uav['y']) / WORLD_SIZE_Y,
                        iot['battery'] / IOT_BATTERY_START,
                        iot['data'] / IOT_DATA_START
                    ])
                else: # Nếu có ít hơn K IoT, đệm bằng số 0
                    closest_iot_states.extend([0, 0, 0, 0])

            # 3. Tìm M vùng cấm gần nhất
            self.no_fly_zones.sort(key=lambda z: np.hypot(z['center'][0] - uav['x'], z['center'][1] - uav['y']))
            closest_nfz_states = []
            for i in range(N_CLOSEST_NFZ):
                if i < len(self.no_fly_zones):
                    zone = self.no_fly_zones[i]
                    closest_nfz_states.extend([
                        (zone['center'][0] - uav['x']) / WORLD_SIZE_X,
                        (zone['center'][1] - uav['y']) / WORLD_SIZE_Y,
                        zone['radius'] / MAX_ZONE_RADIUS
                    ])
                else:
                    closest_nfz_states.extend([0, 0, 0])

            # Nối tất cả lại cho 1 UAV
            single_uav_obs = np.concatenate([uav_state, closest_iot_states, closest_nfz_states])
            all_obs.append(single_uav_obs)
            
        return np.concatenate(all_obs).astype(np.float32)

    def reset(self, seed=None, options=None):
        # Logic reset giữ nguyên như v6, chỉ thay đổi giá trị trả về
        super().reset(seed=seed)
        self.current_step = 0
        self.uavs = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'energy': UAV_ENERGY_START} for _ in range(self.num_uavs)]
        self.iot_nodes = [{'x': self.np_random.uniform(0, WORLD_SIZE_X), 'y': self.np_random.uniform(0, WORLD_SIZE_Y), 'battery': IOT_BATTERY_START, 'data': IOT_DATA_START} for _ in range(self.num_iot_nodes)]
        self.no_fly_zones = []
        for _ in range(NUM_NO_FLY_ZONES):
            center = (self.np_random.uniform(100, WORLD_SIZE_X-100), self.np_random.uniform(100, WORLD_SIZE_Y-100))
            radius = self.np_random.uniform(50, MAX_ZONE_RADIUS)
            self.no_fly_zones.append({'center': center, 'radius': radius})
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Logic hàm step có thể giữ nguyên như v4 (bỏ Reward Shaping để đơn giản hóa)
        # Hoặc giữ nguyên v6 cũng được, nhưng hãy bắt đầu lại từ v4 để xem hiệu quả của Observation
        self.current_step += 1
        reward = 0
        total_throughput = 0; total_energy_consumed = 0
        
        for i in range(self.num_uavs):
            uav = self.uavs[i]
            delta_x, delta_y = action[i*2] * UAV_SPEED, action[i*2 + 1] * UAV_SPEED
            uav['x'] += delta_x; uav['y'] += delta_y
            energy_consumed_this_step = UAV_ENERGY_CONSUMPTION_HOVER if delta_x == 0 and delta_y == 0 else UAV_ENERGY_CONSUMPTION_FLY
            uav['energy'] -= energy_consumed_this_step; total_energy_consumed += energy_consumed_this_step
            if not (0 <= uav['x'] <= WORLD_SIZE_X and 0 <= uav['y'] <= WORLD_SIZE_Y):
                reward -= W_OUT_OF_BOUNDS; uav['x'] = np.clip(uav['x'], 0, WORLD_SIZE_X); uav['y'] = np.clip(uav['y'], 0, WORLD_SIZE_Y)
            for zone in self.no_fly_zones:
                if np.hypot(uav['x'] - zone['center'][0], uav['y'] - zone['center'][1]) < zone['radius']:
                    reward -= W_IN_NO_FLY_ZONE
        
        for iot_node in self.iot_nodes:
            if iot_node['data'] > 0 and iot_node['battery'] > 0:
                for uav in self.uavs:
                    if np.hypot(uav['x'] - iot_node['x'], uav['y'] - iot_node['y']) < COMM_RANGE:
                        data_transmitted = min(iot_node['data'], IOT_DATA_RATE)
                        iot_node['data'] -= data_transmitted; iot_node['battery'] -= IOT_ENERGY_CONSUMPTION_TX
                        total_throughput += data_transmitted
                        break
                        
        reward += total_throughput * W_THROUGHPUT
        reward -= total_energy_consumed * W_UAV_ENERGY
        terminated = all(u['energy'] <= 0 for u in self.uavs) or all(i['data'] <= 0 for i in self.iot_nodes)
        truncated = self.current_step >= MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    # các hàm get_info, close giữ nguyên
    def _get_info(self):
        info = { "uav_positions": [(u['x'], u['y']) for u in self.uavs], "no_fly_zones": self.no_fly_zones }
        info["iot_data_left"] = sum(i['data'] for i in self.iot_nodes)
        return info
    def close(self): pass
