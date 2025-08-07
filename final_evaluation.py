# final_evaluation.py

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Point, LineString

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC 

from stable_baselines3.common.vec_env import DummyVecEnv
from uav_env import UAVNetworkEnv, IOT_DATA_START

# PHẦN 1: CÁC HÀM ĐÁNH GIÁ
def evaluate_rl_agent(model_path, algorithm_class, num_episodes=5):
    """Đánh giá một agent RL đã huấn luyện."""
    algo_name = algorithm_class.__name__
    print(f"\n--- Bắt đầu Đánh giá: {algo_name} Agent ---")
    
    def make_env():
        return UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    eval_env = DummyVecEnv([make_env])

    try:
        model = algorithm_class.load(model_path, env=eval_env, device='cpu')
    except Exception as e:
        print(f"Lỗi khi tải model {algo_name}: {e}"); return None

    rewards, data = [], []
    for episode in range(num_episodes):
        # ======================== SỬA LỖI TRIỆT ĐỂ Ở ĐÂY ========================
        obs = eval_env.reset()
        
        # Lấy thông tin cần thiết trực tiếp từ các thuộc tính của env gốc
        num_uavs = eval_env.get_attr('num_uavs')[0]
        iot_nodes = eval_env.get_attr('iot_nodes')[0]
        uavs = eval_env.get_attr('uavs')[0]
        no_fly_zones = eval_env.get_attr('no_fly_zones')[0]

        iot_positions = [(n['x'], n['y']) for n in iot_nodes]
        trajectories = [[(u['x'], u['y'])] for u in uavs]
        # =======================================================================
        
        done, episode_reward = False, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info_vec = eval_env.step(action)
            episode_reward += reward[0]; done = done_vec[0]
            info = info_vec[0]
            for i, pos in enumerate(info['uav_positions']):
                trajectories[i].append(pos)
        
        rewards.append(episode_reward)
        data.append((eval_env.get_attr('num_iot_nodes')[0] * IOT_DATA_START) - info['iot_data_left'])
    
    avg_reward, avg_data = np.mean(rewards), np.mean(data)
    print(f"-> Kết quả {algo_name}: Reward={avg_reward:.2f}, Data Collected={avg_data:.2f}")
    # Truyền no_fly_zones được lấy lúc đầu vào hàm plot
    plot_trajectory(trajectories, iot_positions, no_fly_zones, f"{algo_name}_Agent")
    return avg_reward, avg_data

def evaluate_baseline(policy_name, policy_function, num_episodes=5):
    # Hàm này đã đúng, không cần sửa
    print(f"\n--- Bắt đầu Đánh giá: {policy_name} Baseline ---")
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    
    rewards, data = [], []
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done, episode_reward = False, 0
        iot_positions = [(n['x'], n['y']) for n in env.iot_nodes]
        no_fly_zones = info['no_fly_zones']
        trajectories = [[pos] for pos in info['uav_positions']]

        while not done:
            action = policy_function(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done, episode_reward = terminated or truncated, episode_reward + reward
            for i, pos in enumerate(info['uav_positions']):
                trajectories[i].append(pos)
        
        rewards.append(episode_reward)
        data.append((env.num_iot_nodes * IOT_DATA_START) - info['iot_data_left'])

    avg_reward, avg_data = np.mean(rewards), np.mean(data)
    print(f"-> Kết quả {policy_name}: Reward={avg_reward:.2f}, Data Collected={avg_data:.2f}")
    plot_trajectory(trajectories, iot_positions, no_fly_zones, f"{policy_name}_Baseline")
    return avg_reward, avg_data

# PHẦN 2: CÁC HÀM HỖ TRỢ (giữ nguyên)
def find_latest_model(algo_name_prefix):
    try:
        models_dir = 'models'
        algo_dirs = [d for d in os.listdir(models_dir) if d.startswith(algo_name_prefix.upper())]
        if not algo_dirs: return None
        algo_dirs.sort(); latest_dir = algo_dirs[-1]
        latest_dir_path = os.path.join(models_dir, latest_dir)
        model_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.zip')]
        if not model_files: return None
        model_files.sort(key=lambda x: int(re.search(r'_(\d+)_steps', x).group(1)) if '_steps' in x else float('inf'))
        return os.path.join(latest_dir_path, model_files[-1])
    except: return None

def plot_trajectory(trajectories, iot_positions, no_fly_zones, file_name):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, zone in enumerate(no_fly_zones):
        ax.add_patch(Circle(zone['center'], zone['radius'], color='red', alpha=0.3, label='No-Fly Zone' if i == 0 else ""))
    iot_x, iot_y = zip(*iot_positions); ax.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    colors = ['darkorange', 'green'];
    for i, path in enumerate(trajectories):
        path_x, path_y = zip(*path); ax.plot(path_x, path_y, color=colors[i], label=f'UAV {i+1} Traj.', zorder=3)
        ax.scatter(path_x[0], path_y[0], c=colors[i], marker='o', s=150, edgecolors='black'); ax.scatter(path_x[-1], path_y[-1], c=colors[i], marker='X', s=200, edgecolors='black')
    ax.set_title(f"Trajectory for {file_name}", fontsize=16); ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.legend(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{file_name}.png"; plt.savefig(fig_path, dpi=300); plt.close()
    print(f"Đã lưu biểu đồ quỹ đạo tại: {fig_path}")

def pathfinding_greedy_policy(env: UAVNetworkEnv):
    actions = []; zones_shapely = [Point(z['center']).buffer(z['radius']) for z in env.no_fly_zones]
    for i in range(env.num_uavs):
        uav = env.uavs[i]; best_target, min_dist = None, float('inf')
        for iot in env.iot_nodes:
            if iot['data'] > 0:
                dist = np.hypot(uav['x'] - iot['x'], uav['y'] - iot['y'])
                if dist < min_dist: min_dist = dist; best_target = iot
        action = np.array([0.0, 0.0])
        if best_target:
            path = LineString([Point(uav['x'], uav['y']), Point(best_target['x'], best_target['y'])])
            is_blocked = any(path.intersects(zone) for zone in zones_shapely)
            direction = np.array([best_target['x'] - uav['x'], best_target['y'] - uav['y']])
            if is_blocked: direction = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(direction); 
            if norm > 0: action = direction / norm
        actions.extend(action)
    return np.array(actions, dtype=np.float32)

def random_policy(env: UAVNetworkEnv): return env.action_space.sample()

# PHẦN 3: CHẠY THỰC NGHIỆM
if __name__ == '__main__':
    # Đổi tên file môi trường v7 để tránh nhầm lẫn
    # Hãy chắc chắn rằng bạn có file uav_env_v7.py
    # Hoặc nếu file môi trường v7 của bạn vẫn tên là uav_env.py thì không cần đổi
    try:
        from uav_env import UAVNetworkEnv, IOT_DATA_START
    except ImportError:
        print("Cảnh báo: Không tìm thấy 'uav_env.py'.")
        exit()

    NUM_EPISODES = 5
    results = {}

    results['Random'] = evaluate_baseline("Random", random_policy, NUM_EPISODES)
    results['Pathfinding_Greedy'] = evaluate_baseline("Pathfinding_Greedy", pathfinding_greedy_policy, NUM_EPISODES)
    
    tqc_model_path = find_latest_model("TQC")
    if tqc_model_path:
        results['TQC'] = evaluate_rl_agent(tqc_model_path, TQC, NUM_EPISODES)
    else: print("\nKhông tìm thấy model TQC.")

    sac_model_path = find_latest_model("SAC")
    if sac_model_path:
        results['SAC'] = evaluate_rl_agent(sac_model_path, SAC, NUM_EPISODES)
    else: print("\nKhông tìm thấy model SAC.")

    print("\n\n" + "="*80)
    print("--- BẢNG TỔNG KẾT THÍ NGHIỆM CUỐI CÙNG (Môi trường v7) ---")
    print("="*80)
    print(f"{'Policy':<25} | {'Avg Reward':<20} | {'Avg Data Collected (%)':<30}")
    print("-" * 80)
    for name, result in sorted(results.items(), key=lambda item: item[1][0] if item[1] else -float('inf'), reverse=True):
        if result:
            reward, data = result
            data_percent = (data / (15 * IOT_DATA_START)) * 100
            print(f"{name:<25} | {reward:<20.2f} | {data:<10.2f} ({data_percent:.1f}%)")
    print("-" * 80)
