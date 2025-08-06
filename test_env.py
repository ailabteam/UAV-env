# test_env.py

from uav_env import UAVNetworkEnv
from gymnasium.utils.env_checker import check_env

# 1. Kiểm tra tính tương thích của môi trường
print("Kiểm tra môi trường với bộ kiểm tra của Gymnasium...")
env_to_check = UAVNetworkEnv()
try:
    check_env(env_to_check)
    print("✅ Môi trường đã vượt qua bài kiểm tra!")
except Exception as e:
    print("❌ Môi trường có lỗi:", e)

# 2. Chạy thử một vài bước với hành động ngẫu nhiên
print("\nChạy thử 10 bước với hành động ngẫu nhiên...")
env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
obs, info = env.reset()
for i in range(10):
    action = env.action_space.sample()  # Lấy một hành động ngẫu nhiên
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Bước {i+1}:")
    print(f"  - Action: {action.round(2)}")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - UAV Positions: {[(round(p[0]), round(p[1])) for p in info['uav_positions']]}")
    print(f"  - Tổng dữ liệu còn lại: {info['iot_data_left']:.2f}")

    if terminated or truncated:
        print("Episode kết thúc!")
        break

env.close()
