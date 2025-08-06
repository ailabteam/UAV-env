# evaluate.py

import os
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import môi trường tùy chỉnh và các hằng số
from uav_env import UAVNetworkEnv, IOT_DATA_START

def evaluate_agent(model_path, num_episodes=1):
    """
    Tải một agent đã được huấn luyện, chạy nó trong môi trường,
    thu thập dữ liệu và lưu kết quả trực quan hóa ra file.
    """
    print(f"--- Đánh giá model từ: {model_path} ---")

    # 1. Tạo môi trường vector hóa
    def make_env():
        return UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    
    eval_env = DummyVecEnv([make_env])

    # 2. Tải model đã huấn luyện
    try:
        model = PPO.load(model_path, env=eval_env, device='cpu')
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return

    total_rewards = []
    total_data_collected = []

    for episode in range(num_episodes):
        print(f"\n--- Bắt đầu Episode #{episode + 1} ---")
        
        # Reset môi trường và lấy trạng thái ban đầu
        # obs có shape (1, obs_dim)
        obs = eval_env.reset()
        
        # ======================== SỬA LỖI LẦN 2 Ở ĐÂY ========================
        # Lấy các thông số cần thiết từ môi trường gốc
        num_uavs = eval_env.get_attr('num_uavs')[0]
        num_iot_nodes = eval_env.get_attr('num_iot_nodes')[0]
        
        trajectories = [[] for _ in range(num_uavs)]
        
        # Lấy vị trí IoT ban đầu.
        iot_nodes_initial = eval_env.get_attr('iot_nodes')[0]
        iot_initial_positions = [(node['x'], node['y']) for node in iot_nodes_initial]
        
        # Lấy vị trí UAV ban đầu.
        uav_initial_positions = eval_env.get_attr('uavs')[0]
        for i in range(num_uavs):
            trajectories[i].append((uav_initial_positions[i]['x'], uav_initial_positions[i]['y']))
        # ================================================================

        done = False
        episode_reward = 0
        current_step = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info_vec = eval_env.step(action)
            
            # Vì là VecEnv, các giá trị trả về là mảng/list.
            # Chúng ta chỉ có 1 môi trường nên lấy phần tử đầu tiên.
            episode_reward += reward[0]
            done = done_vec[0] 
            info = info_vec[0]
            
            current_step += 1

            # Ghi lại vị trí mới từ info trả về của step
            uav_current_positions = info['uav_positions']
            for i in range(num_uavs):
                trajectories[i].append(uav_current_positions[i])
        
        # Lấy thông tin cuối cùng
        data_left = info['iot_data_left']
        initial_data = num_iot_nodes * IOT_DATA_START
        data_collected = initial_data - data_left
        
        total_rewards.append(episode_reward)
        total_data_collected.append(data_collected)
        
        print(f"Episode kết thúc sau {current_step} bước.")
        print(f"Tổng phần thưởng: {episode_reward:.2f}")
        print(f"Tổng dữ liệu thu thập được: {data_collected:.2f} / {initial_data:.2f}")

    print(f"\n--- Kết quả trung bình sau {num_episodes} episode(s) ---")
    print(f"Phần thưởng trung bình: {np.mean(total_rewards):.2f}")
    print(f"Dữ liệu thu thập được trung bình: {np.mean(total_data_collected):.2f}")

    plot_trajectory(trajectories, iot_initial_positions, model_path)


def plot_trajectory(trajectories, iot_positions, model_path):
    # Hàm này không thay đổi
    plt.figure(figsize=(12, 12))
    iot_x = [pos[0] for pos in iot_positions]
    iot_y = [pos[1] for pos in iot_positions]
    plt.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    colors = ['red', 'green', 'purple', 'orange']
    for i, path in enumerate(trajectories):
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        plt.plot(path_x, path_y, color=colors[i % len(colors)], linestyle='-', linewidth=2, label=f'UAV {i+1} Trajectory', alpha=0.8)
        plt.scatter(path_x[0], path_y[0], c=colors[i % len(colors)], marker='o', s=150, edgecolors='black', label=f'UAV {i+1} Start', zorder=6)
        plt.scatter(path_x[-1], path_y[-1], c=colors[i % len(colors)], marker='X', s=200, edgecolors='black', label=f'UAV {i+1} End', zorder=6)
    plt.title(f"Optimized UAV Trajectories\n(Model: {os.path.basename(model_path)})", fontsize=16)
    plt.xlabel("X-coordinate (meters)", fontsize=12)
    plt.ylabel("Y-coordinate (meters)", fontsize=12)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{os.path.basename(model_path).replace('.zip', '')}_trajectory.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nĐã lưu biểu đồ quỹ đạo tại: {fig_path}")
    plt.close()


if __name__ == '__main__':
    # Logic tìm model không đổi
    try:
        models_dir = 'models'
        list_of_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if not list_of_dirs:
            print("Lỗi: Không tìm thấy thư mục model nào. Bạn đã huấn luyện chưa?")
        else:
            list_of_dirs.sort()
            latest_dir = list_of_dirs[-1]
            
            latest_dir_path = os.path.join(models_dir, latest_dir)
            model_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.zip')]
            if not model_files:
                print(f"Lỗi: Không tìm thấy file model .zip nào trong thư mục '{latest_dir_path}'")
            else:
                def get_step_from_filename(filename):
                    base_name = filename.replace('.zip', '')
                    parts = base_name.split('_')
                    for part in reversed(parts):
                        if part.isdigit():
                            return int(part)
                    if 'final' in parts:
                        return float('inf') # Đưa file 'final' lên cuối cùng
                    return -1

                model_files.sort(key=get_step_from_filename)
                latest_model = model_files[-1]

                MODEL_TO_EVALUATE = os.path.join(latest_dir_path, latest_model)
                
                evaluate_agent(model_path=MODEL_TO_EVALUATE, num_episodes=5)
    
    except FileNotFoundError:
        print(f"Lỗi: Thư mục '{models_dir}' không tồn tại. Hãy chắc chắn rằng bạn đã chạy script huấn luyện trước.")
