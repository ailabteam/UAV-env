# master_evaluation.py

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Point, LineString
import copy
import pandas as pd

from stable_baselines3 import SAC
from sb3_contrib import TQC 
from uav_env import UAVNetworkEnv, IOT_DATA_START

# ==============================================================================
# PHẦN 1: CÁC HÀM ĐÁNH GIÁ (LOGIC MỚI, TƯỜNG MINH)
# ==============================================================================

def run_single_episode_rl(model_path, algorithm_class, initial_state):
    """Chạy 1 episode cho agent RL trên một môi trường đơn lẻ."""
    # Tạo một môi trường đơn lẻ
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    # Ép trạng thái ban đầu
    obs, info = env.force_state(initial_state['uavs'], initial_state['iot_nodes'], initial_state['no_fly_zones'])
    
    # Tải model
    model = algorithm_class.load(model_path, env=env, device='cpu')
    
    done, episode_reward = False, 0
    trajectories = [[(u['x'], u['y'])] for u in initial_state['uavs']]
    
    while not done:
        # Quan trọng: obs cần được reshape để có batch dimension
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0]) # Lấy action từ batch
        done = terminated or truncated
        episode_reward += reward
        
        for i, pos in enumerate(info['uav_positions']):
            trajectories[i].append(pos)
            
    final_data_left = sum(iot['data'] for iot in env.iot_nodes)
    data_collected = (env.num_iot_nodes * IOT_DATA_START) - final_data_left
    trajectory_data = {'trajectories': trajectories, 'iot_pos': [(n['x'], n['y']) for n in initial_state['iot_nodes']], 'nfz': initial_state['no_fly_zones']}
    return episode_reward, data_collected, trajectory_data

def run_single_episode_baseline(policy_function, initial_state):
    """Chạy 1 episode cho baseline trên một môi trường đơn lẻ."""
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    env.force_state(initial_state['uavs'], initial_state['iot_nodes'], initial_state['no_fly_zones'])
    
    done, episode_reward = False, 0
    trajectories = [[(u['x'], u['y'])] for u in initial_state['uavs']]
    
    while not done:
        action = policy_function(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        for i, pos in enumerate(info['uav_positions']):
            trajectories[i].append(pos)
            
    final_data_left = sum(iot['data'] for iot in env.iot_nodes)
    data_collected = (env.num_iot_nodes * IOT_DATA_START) - final_data_left
    trajectory_data = {'trajectories': trajectories, 'iot_pos': [(n['x'], n['y']) for n in initial_state['iot_nodes']], 'nfz': initial_state['no_fly_zones']}
    return episode_reward, data_collected, trajectory_data

# ==============================================================================
# PHẦN 2: CÁC HÀM HỖ TRỢ (giữ nguyên)
# ==============================================================================
def find_latest_model(algo_name_prefix):
    try:
        models_dir = 'models'
        algo_dirs = [d for d in os.listdir(models_dir) if d.lower().startswith(algo_name_prefix.lower())]
        if not algo_dirs: print(f"Không tìm thấy thư mục nào bắt đầu bằng '{algo_name_prefix}'"); return None
        algo_dirs.sort(); latest_dir = algo_dirs[-1]
        latest_dir_path = os.path.join(models_dir, latest_dir)
        print(f"Đang tìm model trong thư mục: {latest_dir_path}")
        model_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.zip')]
        if not model_files: print(f"Không có file .zip nào trong {latest_dir_path}"); return None
        final_model = next((f for f in model_files if 'final' in f), None)
        if final_model:
            print(f"Đã tìm thấy model final: {final_model}")
            return os.path.join(latest_dir_path, final_model)
        model_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else -1, reverse=True)
        latest_step_model = model_files[0]
        print(f"Không tìm thấy model final, sử dụng model có bước cao nhất: {latest_step_model}")
        return os.path.join(latest_dir_path, latest_step_model)
    except Exception as e:
        print(f"Lỗi khi tìm model cho '{algo_name_prefix}': {e}"); return None

def plot_trajectory(trajectories, iot_positions, no_fly_zones, file_path):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, zone in enumerate(no_fly_zones):
        ax.add_patch(Circle(zone['center'], zone['radius'], color='red', alpha=0.3, label='No-Fly Zone' if i == 0 else ""))
    iot_x, iot_y = zip(*iot_positions); ax.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    colors = ['darkorange', 'green'];
    for i, path in enumerate(trajectories):
        path_x, path_y = zip(*path); ax.plot(path_x, path_y, color=colors[i], label=f'UAV {i+1} Traj.', zorder=3)
        ax.scatter(path_x[0], path_y[0], c=colors[i], marker='o', s=150, edgecolors='black'); ax.scatter(path_x[-1], path_y[-1], c=colors[i], marker='X', s=200, edgecolors='black')
    ax.set_title(f"Trajectory for {os.path.basename(file_path)}", fontsize=16); ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.legend(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(f"{file_path}.png", dpi=300); plt.close()
    print(f"  -> Đã lưu biểu đồ quỹ đạo tại: {file_path}.png")

def pathfinding_greedy_policy(env: UAVNetworkEnv):
    actions = []; zones_shapely = [Point(z['center']).buffer(z['radius']) for z in env.no_fly_zones]
    for i in range(env.num_uavs):
        uav = env.uavs[i]; best_target = None; min_dist = float('inf')
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

# ==============================================================================
# PHẦN 3: CHẠY THỰC NGHIỆM CHÍNH
# ==============================================================================
if __name__ == '__main__':
    NUM_TOTAL_RUNS = 3
    NUM_EPISODES_PER_RUN = 5
    
    all_results_data = []

    # Định nghĩa các agent và baseline
    policies_to_evaluate = {}
    tqc_path = find_latest_model("TQC_v7"); sac_path = find_latest_model("SAC_v7")
    if tqc_path: policies_to_evaluate['TQC'] = (tqc_path, True, TQC)
    if sac_path: policies_to_evaluate['SAC'] = (sac_path, True, SAC)
    policies_to_evaluate['Random'] = (random_policy, False)
    policies_to_evaluate['Pathfinding_Greedy'] = (pathfinding_greedy_policy, False)

    # Vòng lặp chính chạy nhiều lần
    for run_idx in range(NUM_TOTAL_RUNS):
        print(f"\n{'='*30} BẮT ĐẦU LẦN CHẠY #{run_idx + 1}/{NUM_TOTAL_RUNS} {'='*30}")
        output_dir = f"evaluation_results/run_{run_idx + 1}"
        run_seeds = [i + (run_idx * NUM_EPISODES_PER_RUN) for i in range(NUM_EPISODES_PER_RUN)]
        
        template_env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        
        for episode_idx in range(NUM_EPISODES_PER_RUN):
            print(f"\n--- [Run {run_idx+1}] Đang chạy Episode #{episode_idx+1} ---")
            
            template_env.reset(seed=run_seeds[episode_idx])
            initial_state = {
                'uavs': copy.deepcopy(template_env.uavs),
                'iot_nodes': copy.deepcopy(template_env.iot_nodes),
                'no_fly_zones': copy.deepcopy(template_env.no_fly_zones)
            }
            start_pos_str = ", ".join([f"({u['x']:.0f},{u['y']:.0f})" for u in initial_state['uavs']])
            print(f"  Kịch bản Seed {run_seeds[episode_idx]}: UAVs bắt đầu tại {start_pos_str}")

            for name, policy_info in policies_to_evaluate.items():
                print(f"    -> Đang đánh giá phương pháp: {name}")
                is_rl = policy_info[1]
                if is_rl:
                    r, d, traj = run_single_episode_rl(policy_info[0], policy_info[2], initial_state)
                else:
                    r, d, traj = run_single_episode_baseline(policy_info[0], initial_state)
                
                all_results_data.append({'run': run_idx+1, 'episode': episode_idx+1, 'policy': name, 'reward': r, 'data': d})
                if episode_idx == NUM_EPISODES_PER_RUN - 1: 
                    plot_trajectory(traj['trajectories'], traj['iot_pos'], traj['nfz'], f"{output_dir}/{name}")

    # --- In bảng tổng kết cuối cùng ---
    print("\n\n" + "="*80)
    print("--- BẢNG TỔNG KẾT THÍ NGHIỆM CUỐI CÙNG (ĐỒNG BỘ HÓA HOÀN TOÀN) ---")
    print("="*80)
    
    df = pd.DataFrame(all_results_data)
    summary = df.groupby('policy').agg(
        avg_reward=('reward', 'mean'), std_reward=('reward', 'std'),
        avg_data=('data', 'mean'), std_data=('data', 'std')
    ).sort_values(by='avg_reward', ascending=False)
    summary['data_percent'] = (summary['avg_data'] / (15 * IOT_DATA_START)) * 100

    print(f"{'Policy':<25} | {'Avg Reward (± std)':<25} | {'Avg Data Collected (%)':<30}")
    print("-" * 90)
    for name, row in summary.iterrows():
        reward_str = f"{row['avg_reward']:.2f} (± {row['std_reward']:.2f})"
        data_str = f"{row['avg_data']:.2f} ({row['data_percent']:.1f}%)"
        print(f"{name:<25} | {reward_str:<25} | {data_str:<30}")
    print("-" * 90)
