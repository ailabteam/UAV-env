# train.py

import os
from datetime import datetime
import torch

# ======================== THAY ĐỔI LỚN ========================
from stable_baselines3 import SAC # Thay PPO bằng SAC
# =============================================================

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Sử dụng môi trường v7 với Feature Engineering
from uav_env import UAVNetworkEnv 

# --- CẤU HÌNH HUẤN LUYỆN ---
TRAIN_TIMESTEPS = 2_000_000
NUM_ENVS = 8 # SAC cũng hoạt động tốt với môi trường song song
LOG_INTERVAL = 1
CHECKPOINT_FREQ = 50_000

# SAC thường không cần nhiều tinh chỉnh như PPO, nhưng đây là một vài tham số tốt
SAC_HYPERPARAMS = {
    "learning_rate": 3e-4, # learning rate mặc định của SAC, thường hoạt động tốt
    "buffer_size": 300_000, # Kích thước Replay Buffer. Tăng nếu có nhiều RAM.
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": (1, "step"), # Cập nhật sau mỗi bước
    "gradient_steps": 1,
}

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Đặt tên thư mục rõ ràng để biết chúng ta đang dùng SAC
    log_dir = f"logs/SAC_{current_time}/"
    model_dir = f"models/SAC_{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        # Vẫn sử dụng môi trường v7 tốt nhất của chúng ta
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        env = Monitor(env)
        return env

    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_sac_model" # Đổi tên model
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    # ======================== THAY ĐỔI LỚN ========================
    # Khởi tạo model SAC thay vì PPO
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        **SAC_HYPERPARAMS # Truyền các siêu tham số của SAC
    )
    # =============================================================

    print("--- Bắt đầu Huấn luyện với Thuật toán SAC ---")
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=LOG_INTERVAL,
        progress_bar=True
    )

    final_model_path = f"{model_dir}/uav_sac_model_final.zip"
    model.save(final_model_path)
    print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")

    vec_env.close()
