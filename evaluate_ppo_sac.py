# evaluate_ppo_sac.py

import os
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import cả PPO và SAC
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.vec_env import DummyVecEnv
from uav_env import UAVNetworkEnv, IOT_DATA_START

def evaluate_rl_agent(model_path, algorithm, num_episodes=5):
    """
    Tải một agent RL (PPO hoặc SAC) đã được huấn luyện, chạy nó,
    thu thập dữ liệu và lưu kết quả.
    """
    print(f"--- Đánh giá model {algorithm.__name__} từ: {model_path} ---")

    def make_env():
        return UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    
    eval_env = DummyVecEnv([make_env])

    try:
        # Tải model dựa trên thuật toán được chỉ định
        model = algorithm.load(model_path, env=eval_env, device='cpu')
    except Exception as e:
        print(f"Lỗi khi tải model: {e}"); return

    total_rewards, total_data_collected = [], []

    for episode in range(num_episodes):
        print(f"\n--- Bắt đầu Episode #{episode + 1} ---")
        obs = eval_env.reset() # Môi trường con sẽ tự dùng seed tuần tự
        
        num_uavs = eval_env.get_attr('num_uavs')[0]
        num_iot_nodes = eval_env.get_attr('num_iot_nodes')[0]
        
        trajectories = [[] for _ in range(num_uavs)]
        
        iot_nodes_initial = eval_env.get_attr('iot_nodes')[0]
        iot_initial_positions = [(n['x'], n['y']) for n in iot_nodes_initial]
        
        uav_initial_positions = eval_env.get_attr('uavs')[0]
        for i in range(num_uavs):
            trajectories[i].append((uav_initial_positions[i]['x'], uav_initial_positions[i]['y']))

        done, episode_reward, current_step = False, 0, 0
        no_fly_zones = eval_env.get_attr('no_fly_zones')[0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info_vec = eval_env.step(action)
            
            episode_reward += reward[0]
            done = done_vec[0]
            info = info_vec[0]
            current_step += 1
            
            for i in range(num_uavs):
                trajectories[i].append(info['uav_positions'][i])
        
        data_collected = (num_iot_nodes * IOT_DATA_START) - info['iot_data_left']
        total_rewards.append(episode_reward)
        total_data_collected.append(data_collected)
        
        print(f"Episode kết thúc sau {current_step} bước.")
        print(f"Tổng phần thưởng: {episode_reward:.2f}\nTổng dữ liệu thu thập: {data_collected:.2f}")

    print(f"\n--- Kết quả trung bình {algorithm.__name__} sau {num_episodes} episode(s) ---")
    print(f"Phần thưởng: {np.mean(total_rewards):.2f}\nDữ liệu thu thập: {np.mean(total_data_collected):.2f}")
    
    plot_trajectory(trajectories, iot_initial_positions, no_fly_zones, model_path, algorithm.__name__)


def plot_trajectory(trajectories, iot_positions, no_fly_zones, model_path, algo_name):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for i, zone in enumerate(no_fly_zones):
        label = 'No-Fly Zone' if i == 0 else ""
        ax.add_patch(Circle(zone['center'], zone['radius'], color='red', alpha=0.3, label=label))
        
    iot_x, iot_y = zip(*iot_positions)
    ax.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    
    colors = ['darkorange', 'green']
    for i, path in enumerate(trajectories):
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color=colors[i], label=f'UAV {i+1} Traj.', zorder=3)
        ax.scatter(path_x[0], path_y[0], c=colors[i], marker='o', s=150, edgecolors='black', label=f'UAV {i+1} Start', zorder=6)
        ax.scatter(path_x[-1], path_y[-1], c=colors[i], marker='X', s=200, edgecolors='black', label=f'UAV {i+1} End', zorder=6)
        
    ax.set_title(f"Optimized Trajectory ({algo_name} Agent)", fontsize=16)
    ax.set_xlabel("X-coordinate (meters)"); ax.set_ylabel("Y-coordinate (meters)")
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.legend(loc='best'); ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{algo_name}_{os.path.basename(model_path).replace('.zip', '')}_trajectory.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nĐã lưu biểu đồ quỹ đạo tại: {fig_path}")
    plt.close()

def find_latest_model(algo_name):
    """Hàm tìm model mới nhất cho một thuật toán cụ thể."""
    try:
        models_dir = 'models'
        # Tìm các thư mục con có tên bắt đầu bằng tên thuật toán (viết hoa)
        algo_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith(algo_name.upper())]
        if not algo_dirs: return None
        
        algo_dirs.sort()
        latest_dir = algo_dirs[-1]
        latest_dir_path = os.path.join(models_dir, latest_dir)
        
        model_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.zip')]
        if not model_files: return None
        
        # Sắp xếp để tìm model final hoặc model có số bước lớn nhất
        model_files.sort(key=lambda x: int(x.split('_')[-2].replace('steps','')) if 'steps' in x else float('inf'))
        latest_model = model_files[-1]
        return os.path.join(latest_dir_path, latest_model)
    except Exception:
        return None

if __name__ == '__main__':
    NUM_EVAL_EPISODES = 5
    
    # --- Đánh giá SAC ---
    sac_model_path = find_latest_model("SAC")
    if sac_model_path:
        evaluate_rl_agent(model_path=sac_model_path, algorithm=SAC, num_episodes=NUM_EVAL_EPISODES)
    else:
        print("Không tìm thấy model SAC để đánh giá.")
        
    # --- Đánh giá PPO (để so sánh sự thất bại) ---
    ppo_model_path = find_latest_model("PPO")
    if ppo_model_path:
        # PPO đã được huấn luyện với môi trường v7, chúng ta cũng có thể đánh giá nó
        # evaluate_rl_agent(model_path=ppo_model_path, algorithm=PPO, num_episodes=NUM_EVAL_EPISODES)
        print("\nBỏ qua đánh giá PPO vì đã biết kết quả không tốt.")
        print("Để chạy, hãy bỏ bình luận dòng code phía trên.")
    else:
        print("\nKhông tìm thấy model PPO để đánh giá.")
