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

# ======================== ĐÃ SỬA LỖI IMPORT ========================
from stable_baselines3 import SAC
from sb3_contrib import TQC
# ====================================================================

from stable_baselines3.common.vec_env import DummyVecEnv
from uav_env import UAVNetworkEnv, IOT_DATA_START

# PHẦN 1: CÁC HÀM ĐÁNH GIÁ (giữ nguyên)
def run_single_episode_rl(model_path, algorithm_class, initial_state):
    def make_env():
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        env.uavs = copy.deepcopy(initial_state['uavs'])
        env.iot_nodes = copy.deepcopy(initial_state['iot_nodes'])
        env.no_fly_zones = copy.deepcopy(initial_state['no_fly_zones'])
        return env
    eval_env = DummyVecEnv([make_env])
    model = algorithm_class.load(model_path, env=eval_env, device='cpu')
    obs = eval_env.reset()
    done, episode_reward = False, 0
    trajectories = [[(u['x'], u['y'])] for u in initial_state['uavs']]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, info_vec = eval_env.step(action)
        episode_reward += reward[0]; done = done_vec[0]
        for i, pos in enumerate(info_vec[0]['uav_positions']):
            trajectories[i].append(pos)
    final_data_left = sum(iot['data'] for iot in eval_env.get_attr('iot_nodes')[0])
    data_collected = (eval_env.get_attr('num_iot_nodes')[0] * IOT_DATA_START) - final_data_left
    trajectory_data = {'trajectories': trajectories, 'iot_pos': [(n['x'], n['y']) for n in initial_state['iot_nodes']], 'nfz': initial_state['no_fly_zones']}
    return episode_reward, data_collected, trajectory_data

def run_single_episode_baseline(policy_function, initial_state):
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    env.uavs = copy.deepcopy(initial_state['uavs'])
    env.iot_nodes = copy.deepcopy(initial_state['iot_nodes'])
    env.no_fly_zones = copy.deepcopy(initial_state['no_fly_zones'])
    obs, info = env.reset()
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

# PHẦN 2: CÁC HÀM HỖ TRỢ (giữ nguyên)
def find_latest_model(algo_name_prefix):
    try:
        models_dir = 'models'; algo_dirs = [d for d in os.listdir(models_dir) if d.startswith(algo_name_prefix.upper())]
        if not algo_dirs: return None
        algo_dirs.sort(); latest_dir = algo_dirs[-1]; latest_dir_path = os.path.join(models_dir, latest_dir)
        model_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.zip')]
        if not model_files: return None
        model_files.sort(key=lambda x: int(re.search(r'_(\d+)_steps', x).group(1)) if '_steps' in x else float('inf'))
        return os.path.join(latest_dir_path, model_files[-1])
    except: return None

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
    print(f"Đã lưu biểu đồ quỹ đạo tại: {file_path}.png")

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

# PHẦN 3: CHẠY THỰC NGHIỆM CHÍNH (giữ nguyên)
if __name__ == '__main__':
    NUM_TOTAL_RUNS = 3
    NUM_EPISODES_PER_RUN = 5
    all_results = []
    rl_models_to_evaluate = {}
    tqc_path = find_latest_model("TQC"); sac_path = find_latest_model("SAC")
    if tqc_path: rl_models_to_evaluate['TQC'] = (tqc_path, TQC)
    if sac_path: rl_models_to_evaluate['SAC'] = (sac_path, SAC)
    baselines_to_evaluate = {'Random': random_policy, 'Pathfinding_Greedy': pathfinding_greedy_policy}

    for run_idx in range(NUM_TOTAL_RUNS):
        print(f"\n{'='*30} BẮT ĐẦU LẦN CHẠY #{run_idx + 1}/{NUM_TOTAL_RUNS} {'='*30}")
        output_dir = f"evaluation_results/run_{run_idx + 1}"
        evaluation_seeds = [i + (run_idx * NUM_EPISODES_PER_RUN) for i in range(NUM_EPISODES_PER_RUN)]
        template_env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        
        for name, policy_func in baselines_to_evaluate.items():
            print(f"\n--- [Run {run_idx+1}] Đang đánh giá: {name} ---")
            rewards, data = [], []
            for i in range(NUM_EPISODES_PER_RUN):
                template_env.reset(seed=evaluation_seeds[i])
                initial_state = {'uavs': copy.deepcopy(template_env.uavs), 'iot_nodes': copy.deepcopy(template_env.iot_nodes), 'no_fly_zones': copy.deepcopy(template_env.no_fly_zones)}
                r, d, traj_data = run_single_episode_baseline(policy_func, initial_state)
                rewards.append(r); data.append(d)
                if i == NUM_EPISODES_PER_RUN - 1:
                    plot_trajectory(traj_data['trajectories'], traj_data['iot_pos'], traj_data['nfz'], f"{output_dir}/{name}_Baseline")
            all_results.append({'run': run_idx+1, 'policy': name, 'reward': np.mean(rewards), 'data': np.mean(data)})

        for name, (model_path, algo_class) in rl_models_to_evaluate.items():
            print(f"\n--- [Run {run_idx+1}] Đang đánh giá: {name} ---")
            rewards, data = [], []
            for i in range(NUM_EPISODES_PER_RUN):
                template_env.reset(seed=evaluation_seeds[i])
                initial_state = {'uavs': copy.deepcopy(template_env.uavs), 'iot_nodes': copy.deepcopy(template_env.iot_nodes), 'no_fly_zones': copy.deepcopy(template_env.no_fly_zones)}
                r, d, traj_data = run_single_episode_rl(model_path, algo_class, initial_state)
                rewards.append(r); data.append(d)
                if i == NUM_EPISODES_PER_RUN - 1:
                    plot_trajectory(traj_data['trajectories'], traj_data['iot_pos'], traj_data['nfz'], f"{output_dir}/{name}_Agent")
            all_results.append({'run': run_idx+1, 'policy': name, 'reward': np.mean(rewards), 'data': np.mean(data)})
            
    print("\n\n" + "="*80)
    print("--- BẢNG TỔNG KẾT THÍ NGHIỆM CUỐI CÙNG (TRUNG BÌNH QUA CÁC LẦN CHẠY) ---")
    print("="*80)
    df = pd.DataFrame(all_results)
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
