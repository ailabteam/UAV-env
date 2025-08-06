# evaluate_baselines.py (Phiên bản cuối cùng)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Point, LineString

from uav_env import UAVNetworkEnv, IOT_DATA_START

def run_evaluation(policy_name, policy_function, num_episodes=5):
    print(f"\n--- Đánh giá Baseline: {policy_name} ---")
    env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
    
    total_rewards, total_data_collected = [], []
    for episode in range(num_episodes):
        # Sử dụng seed tuần tự để so sánh công bằng với các agent RL
        obs, info = env.reset(seed=episode)
        
        done, episode_reward = False, 0
        trajectories = [[info['uav_positions'][i]] for i in range(env.num_uavs)]
        iot_initial_positions = [(n['x'], n['y']) for n in env.iot_nodes]
        no_fly_zones = info['no_fly_zones']

        while not done:
            action = policy_function(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            for i in range(env.num_uavs):
                trajectories[i].append(info['uav_positions'][i])

        data_collected = (env.num_iot_nodes * IOT_DATA_START) - info['iot_data_left']
        total_rewards.append(episode_reward)
        total_data_collected.append(data_collected)
        print(f"  Episode #{episode+1}: Reward={episode_reward:.2f}, Data Collected={data_collected:.2f}")

    avg_reward, avg_data = np.mean(total_rewards), np.mean(total_data_collected)
    print(f"--- Kết quả trung bình cho {policy_name} ---\nPhần thưởng: {avg_reward:.2f}\nDữ liệu thu thập được: {avg_data:.2f}")
    
    plot_trajectory(trajectories, iot_initial_positions, no_fly_zones, f"{policy_name}_trajectory")
    return avg_reward, avg_data

def pathfinding_greedy_policy(env: UAVNetworkEnv):
    actions = []
    # Chuyển đổi vùng cấm sang đối tượng shapely để kiểm tra giao cắt
    zones_shapely = [Point(z['center']).buffer(z['radius']) for z in env.no_fly_zones]

    for i in range(env.num_uavs):
        uav = env.uavs[i]
        best_target, min_dist = None, float('inf')
        for iot in env.iot_nodes:
            if iot['data'] > 0:
                dist = np.hypot(uav['x'] - iot['x'], uav['y'] - iot['y'])
                if dist < min_dist:
                    min_dist = dist; best_target = iot
        
        action = np.array([0.0, 0.0])
        if best_target:
            start_point = Point(uav['x'], uav['y'])
            end_point = Point(best_target['x'], best_target['y'])
            direct_path = LineString([start_point, end_point])
            
            is_path_blocked = any(direct_path.intersects(zone) for zone in zones_shapely)
            
            direction_vector = np.array([best_target['x'] - uav['x'], best_target['y'] - uav['y']])
            if is_path_blocked:
                # Heuristic né: xoay vector hướng 90 độ
                direction_vector = np.array([-direction_vector[1], direction_vector[0]])
            
            norm = np.linalg.norm(direction_vector)
            if norm > 0: action = direction_vector / norm
        actions.extend(action)
        
    return np.array(actions, dtype=np.float32)

def random_policy(env: UAVNetworkEnv):
    return env.action_space.sample()

def plot_trajectory(trajectories, iot_positions, no_fly_zones, policy_name):
    # Hàm này không cần thay đổi nhiều, đã tốt
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, zone in enumerate(no_fly_zones):
        ax.add_patch(Circle(zone['center'], zone['radius'], color='red', alpha=0.3, label='No-Fly Zone' if i == 0 else ""))
    iot_x, iot_y = zip(*iot_positions); ax.scatter(iot_x, iot_y, c='blue', marker='s', s=100, label='IoT Nodes', zorder=5)
    colors = ['darkorange', 'green'];
    for i, path in enumerate(trajectories):
        path_x, path_y = zip(*path); ax.plot(path_x, path_y, color=colors[i], label=f'UAV {i+1} Traj.', zorder=3)
        ax.scatter(path_x[0], path_y[0], c=colors[i], marker='o', s=150, edgecolors='black'); ax.scatter(path_x[-1], path_y[-1], c=colors[i], marker='X', s=200, edgecolors='black')
    ax.set_title(f"Trajectory for {policy_name}", fontsize=16); ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.legend(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    os.makedirs("evaluation_results", exist_ok=True)
    fig_path = f"evaluation_results/{policy_name}.png"; plt.savefig(fig_path, dpi=300); plt.close()
    print(f"Đã lưu biểu đồ quỹ đạo tại: {fig_path}")

if __name__ == '__main__':
    NUM_EVAL_EPISODES = 5
    results = {}
    
    results['Pathfinding_Greedy'] = run_evaluation("Pathfinding_Greedy", pathfinding_greedy_policy, num_eval_episodes)
    results['Random'] = run_evaluation("Random", random_policy, num_eval_episodes)
    
    print("\n\n--- BẢNG TỔNG KẾT SO SÁNH BASELINES ---")
    print(f"{'Policy':<25} | {'Avg Reward':<15} | {'Avg Data Collected':<20}")
    print("-" * 65)
    for name, (reward, data) in results.items():
        print(f"{name:<25} | {reward:<15.2f} | {data:<20.f}")
    print("-" * 65)
