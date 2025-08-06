# evaluate.py

import os
import gymnasium as gym
import numpy as np
import matplotlib
# ======================== THAY ĐỔI ========================
# Sử dụng một backend không yêu cầu giao diện đồ họa
matplotlib.use('Agg') 
# ========================================================
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import môi trường tùy chỉnh và các hằng số
from uav_env import UAVNetworkEnv, MAX_STEPS

def evaluate_agent(model_path, num_episodes=1):
    """
    Tải một agent đã được huấn luyện, chạy nó trong môi trường,
    thu thập dữ liệu và lưu kết quả trực quan hóa ra file.
    """
    print(f"--- Đánh giá model từ: {model_path} ---")

    # 1. Tải môi trường
    eval_env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)

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
        
        trajectories = [[] for _ in range(eval_env.num_uavs)]
        iot_initial_positions = [(node['x'], node['y']) for node in eval_env.iot_nodes]
        
        obs, info = eval_env.reset(seed=episode) # Dùng seed để kết quả có thể tái tạo
        done = False
        episode_reward = 0
        
        for i in range(eval_env.num_uavs):
            trajectories[i].append(info['uav_positions'][i])

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward

            for i in range(eval_env.num_uavs):
                trajectories[i].append(info['uav_positions'][i])

        total_rewards.append(episode_reward)
        data_left = info['iot_data_left']
        initial_data = eval_env.num_iot_nodes * 100.0
        data_collected = initial_data - data_left
        total_data_collected.append(data_collected)
        
        print(f"Episode kết thúc sau {eval_env.current_step} bước.")
        print(f"Tổng phần thưởng: {episode_reward:.2f}")
        print(f"Tổng dữ liệu thu thập được: {data_collected:.2f} / {initial_data:.2f}")

    print(f"\n--- Kết quả trung bình sau {num_episodes} episode(s) ---")
    print(f"Phần thưởng trung bình: {np.mean(total_rewards):.2f}")
    print(f"Dữ liệu thu thập được trung bình: {np.mean(total_data_collected):.2f}")

    # Trực quan hóa kết quả của episode cuối cùng
    plot_trajectory(trajectories, iot_initial_positions, model_path)


def plot_trajectory(trajectories, iot_positions, model_path):
    """
    Vẽ quỹ đạo của các UAV và vị trí của các node IoT, sau đó lưu ra file.
    """
    plt.figure(figsize=(12, 12))
    
    iot_x = [pos[0] for pos in iot_positions]
    iot_y = [pos[1] for pos in iot_positions]
    plt.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)

    colors = ['red', 'green', 'purple', 'orange']
    for i, path in enumerate(trajectories):
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        plt.plot(path_x, path_y, color=colors[i % len(colors)], linestyle='-', 
                 linewidth=2, label=f'UAV {i+1} Trajectory', alpha=0.8)
        
        plt.scatter(path_x[0], path_y[0], c=colors[i % len(colors)], marker='o', 
                    s=150, edgecolors='black', label=f'UAV {i+1} Start', zorder=6)
        plt.scatter(path_x[-1], path_y[-1], c=colors[i % len(colors)], marker='X', 
                    s=200, edgecolors='black', label=f'UAV {i+1} End', zorder=6)

    plt.title(f"Optimized UAV Trajectories\n(Model: {os.path.basename(model_path)})", fontsize=16)
    plt.xlabel("X-coordinate (meters)", fontsize=12)
    plt.ylabel("Y-coordinate (meters)", fontsize=12)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')

    # Lưu hình ảnh
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{os.path.basename(model_path).replace('.zip', '')}_trajectory.png"
    plt.savefig(fig_path, dpi=300) # Lưu với độ phân giải cao cho paper
    print(f"\nĐã lưu biểu đồ quỹ đạo tại: {fig_path}")
    
    # ======================== THAY ĐỔI ========================
    # Đóng figure để giải phóng bộ nhớ và không cố gắng hiển thị nó
    plt.close()
    # plt.show() # Bình luận dòng này để tránh lỗi trên server
    # ========================================================

if __name__ == '__main__':
    # Đoạn code này sẽ tự động tìm model mới nhất đã được huấn luyện
    try:
        list_of_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))]
        if not list_of_dirs:
            print("Lỗi: Không tìm thấy thư mục model nào. Bạn đã huấn luyện chưa?")
        else:
            list_of_dirs.sort()
            latest_dir = list_of_dirs[-1]
            
            model_files = [f for f in os.listdir(f'models/{latest_dir}') if f.endswith('.zip')]
            if not model_files:
                print(f"Lỗi: Không tìm thấy file model .zip nào trong thư mục 'models/{latest_dir}'")
            else:
                # Sắp xếp để tìm model có số bước lớn nhất hoặc model final
                model_files.sort(key=lambda x: int(x.split('_')[-2].replace('steps','')) if 'steps' in x else float('inf'))
                latest_model = model_files[-1]

                MODEL_TO_EVALUATE = os.path.join("models", latest_dir, latest_model)
                
                evaluate_agent(model_path=MODEL_TO_EVALUATE, num_episodes=5)
    
    except FileNotFoundError:
        print("Lỗi: Thư mục 'models' không tồn tại. Hãy chắc chắn rằng bạn đã chạy script huấn luyện trước.")
