# evaluate_baselines.py

import os
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uav_env import UAVNetworkEnv, IOT_DATA_START

def run_evaluation(policy_name, policy_function, num_episodes=5):
    """
    Hàm chung để chạy đánh giá cho một chính sách (policy) nhất định.
    """
    print(f"\n--- Đánh giá Baseline: {policy_name} ---")
    
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    
    total_rewards = []
    total_data_collected = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode) # Dùng cùng seed với evaluate.py để so sánh công bằng
        done = False
        episode_reward = 0
        current_step = 0

        # Lưu quỹ đạo
        trajectories = [[] for _ in range(env.num_uavs)]
        for i in range(env.num_uavs):
            trajectories[i].append(info['uav_positions'][i])
            
        iot_initial_positions = [(node['x'], node['y']) for node in env.iot_nodes]

        while not done:
            # Lấy hành động từ hàm chính sách của baseline
            action = policy_function(env)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            current_step += 1
            
            for i in range(env.num_uavs):
                trajectories[i].append(info['uav_positions'][i])

        # Tính toán kết quả
        data_left = info['iot_data_left']
        initial_data = env.num_iot_nodes * IOT_DATA_START
        data_collected = initial_data - data_left
        
        total_rewards.append(episode_reward)
        total_data_collected.append(data_collected)
        
        print(f"  Episode #{episode+1}: Reward={episode_reward:.2f}, Data Collected={data_collected:.2f}")

    avg_reward = np.mean(total_rewards)
    avg_data = np.mean(total_data_collected)
    
    print(f"--- Kết quả trung bình cho {policy_name} ---")
    print(f"Phần thưởng trung bình: {avg_reward:.2f}")
    print(f"Dữ liệu thu thập được trung bình: {avg_data:.2f}")
    
    # Vẽ quỹ đạo cho episode cuối
    plot_trajectory(trajectories, iot_initial_positions, f"{policy_name}_trajectory")
    
    return avg_reward, avg_data

# --- ĐỊNH NGHĨA CÁC HÀM CHÍNH SÁCH (POLICY FUNCTIONS) ---

def greedy_policy(env: UAVNetworkEnv):
    """
    Chính sách tham lam: mỗi UAV bay về phía node IoT chưa được phục vụ gần nhất.
    """
    actions = []
    for i in range(env.num_uavs):
        uav = env.uavs[i]
        
        best_target = None
        min_dist = float('inf')
        
        # Tìm node IoT "hấp dẫn" nhất (còn dữ liệu)
        for iot in env.iot_nodes:
            if iot['data'] > 0:
                dist = np.sqrt((uav['x'] - iot['x'])**2 + (uav['y'] - iot['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_target = iot
        
        if best_target:
            # Tính vector hướng đến mục tiêu và chuẩn hóa
            direction_vector = np.array([best_target['x'] - uav['x'], best_target['y'] - uav['y']])
            norm = np.linalg.norm(direction_vector)
            if norm > 0:
                action = direction_vector / norm
            else:
                action = np.array([0.0, 0.0]) # Đã đến nơi
        else:
            action = np.array([0.0, 0.0]) # Không còn mục tiêu
        
        actions.extend(action)
        
    return np.array(actions, dtype=np.float32)


def fixed_trajectory_policy(env: UAVNetworkEnv):
    """
    Chính sách quỹ đạo cố định: UAV 1 bay theo vòng tròn, UAV 2 bay zig-zag.
    Đây là một ví dụ, có thể thiết kế các quỹ đạo khác.
    """
    actions = []
    
    # UAV 1: Bay theo vòng tròn quanh trung tâm
    uav1 = env.uavs[0]
    center_x, center_y = 500, 500
    radius = 350
    # Tính góc hiện tại so với trung tâm
    angle = np.arctan2(uav1['y'] - center_y, uav1['x'] - center_x)
    # Hướng di chuyển tiếp theo là tiếp tuyến với vòng tròn (vuông góc với bán kính)
    delta_x1 = -np.sin(angle)
    delta_y1 = np.cos(angle)
    actions.extend([delta_x1, delta_y1])
    
    # UAV 2: Bay theo đường zig-zag đơn giản
    uav2 = env.uavs[1]
    # Nếu đang ở gần biên trái/phải, đổi hướng y
    if 'direction_y' not in uav2: uav2['direction_y'] = 1 
    if uav2['x'] > 950 or uav2['x'] < 50:
        uav2['direction_y'] *= -1
    
    # Luôn bay sang phải
    delta_x2 = 1.0
    delta_y2 = 0.5 * uav2['direction_y']
    action2_unnormalized = np.array([delta_x2, delta_y2])
    actions.extend(action2_unnormalized / np.linalg.norm(action2_unnormalized))
    
    return np.array(actions, dtype=np.float32)

def random_policy(env: UAVNetworkEnv):
    """Chính sách ngẫu nhiên để làm baseline thấp nhất."""
    return env.action_space.sample()

def plot_trajectory(trajectories, iot_positions, policy_name):
    # Hàm này gần giống hàm trong evaluate.py
    plt.figure(figsize=(12, 12))
    iot_x = [pos[0] for pos in iot_positions]
    iot_y = [pos[1] for pos in iot_positions]
    plt.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    colors = ['red', 'green']
    for i, path in enumerate(trajectories):
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color=colors[i], label=f'UAV {i+1} ({policy_name})')
        plt.scatter(path_x[0], path_y[0], c=colors[i], marker='o', s=150, edgecolors='black')
        plt.scatter(path_x[-1], path_y[-1], c=colors[i], marker='X', s=200, edgecolors='black')
    plt.title(f"Trajectory for {policy_name}", fontsize=16)
    plt.xlim(0, 1000); plt.ylim(0, 1000); plt.legend(); plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{policy_name}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Đã lưu biểu đồ quỹ đạo tại: {fig_path}")
    plt.close()

if __name__ == '__main__':
    num_eval_episodes = 5
    
    # Chạy và lưu kết quả cho từng baseline
    results = {}
    results['Greedy'] = run_evaluation("Greedy", greedy_policy, num_eval_episodes)
    results['Fixed_Trajectory'] = run_evaluation("Fixed_Trajectory", fixed_trajectory_policy, num_eval_episodes)
    results['Random'] = run_evaluation("Random", random_policy, num_eval_episodes)
    
    print("\n\n--- BẢNG TỔNG KẾT SO SÁNH ---")
    print("--------------------------------------------------")
    print(f"{'Policy':<20} | {'Avg Reward':<15} | {'Avg Data Collected':<20}")
    print("--------------------------------------------------")
    for name, (reward, data) in results.items():
        print(f"{name:<20} | {reward:<15.2f} | {data:<20.2f}")
    print("--------------------------------------------------")
    print("Ghi chú: Hãy thêm kết quả của PPO Agent vào bảng này để so sánh!")
