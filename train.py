# train.py

import os
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
# ======================== THÊM MỚI ========================
from stable_baselines3.common.monitor import Monitor
# ==========================================================

# Import môi trường tùy chỉnh của chúng ta
from uav_env import UAVNetworkEnv

# --- CẤU HÌNH HUẤN LUYỆN ---
TRAIN_TIMESTEPS = 2_000_000
NUM_ENVS = 8
LOG_INTERVAL = 1
CHECKPOINT_FREQ = 50_000

if __name__ == '__main__':
    # 1. Thiết lập thư mục lưu trữ
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{current_time}/"
    model_dir = f"models/{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Tạo các môi trường song song (Vectorized Environment)
    # ======================== THAY ĐỔI Ở ĐÂY ========================
    def make_env():
        # Định nghĩa môi trường gốc
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        # Bọc nó trong Monitor wrapper để theo dõi thông tin episode
        # Monitor sẽ tạo ra một file .csv chứa thông tin reward, length,...
        # và quan trọng là nó sẽ đưa thông tin này vào log của TensorBoard.
        env = Monitor(env)
        return env

    # SubprocVecEnv sẽ nhận vào hàm tạo môi trường đã được bọc Monitor
    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])
    # ================================================================

    # 3. Tạo Callback (giữ nguyên)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_ppo_model"
    )

    # 4. Định nghĩa AI Agent (giữ nguyên)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=log_dir
    )

    # 5. Bắt đầu quá trình huấn luyện (giữ nguyên)
    print("Bắt đầu huấn luyện AI Agent...")
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=LOG_INTERVAL,
        progress_bar=True
    )

    # 6. Lưu model cuối cùng (giữ nguyên)
    final_model_path = f"{model_dir}/uav_ppo_model_final.zip"
    model.save(final_model_path)
    print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")

    vec_env.close()
