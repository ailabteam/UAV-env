# train_curriculum_phase1.py

import os
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Import môi trường đã được nâng cấp
from uav_env import UAVNetworkEnv

# --- CẤU HÌNH ---
# Chúng ta không cần huấn luyện quá lâu ở giai đoạn này
TRAIN_TIMESTEPS_PHASE1 = 1_000_000 
NUM_ENVS = 8
# Dùng lại các siêu tham số đã tinh chỉnh
PPO_HYPERPARAMS = {
    "n_steps": 8192, "batch_size": 512, "gamma": 0.99,
    "learning_rate": 1e-4, "ent_coef": 0.01, "clip_range": 0.2,
    "n_epochs": 10, "gae_lambda": 0.95,
}

if __name__ == '__main__':
    # Thiết lập thư mục riêng cho giai đoạn 1
    log_dir = "logs/curriculum_phase1/"
    model_dir = "models/curriculum_phase1/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        # ======================== GỌI MÔI TRƯỜNG Ở CHẾ ĐỘ 'simple' ========================
        env = UAVNetworkEnv(num_uavs=2, mode='simple')
        env = Monitor(env)
        return env

    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_phase1_model"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    model = PPO(
        policy="MlpPolicy", env=vec_env, device=device,
        verbose=1, tensorboard_log=log_dir, **PPO_HYPERPARAMS
    )

    print("--- Bắt đầu Huấn luyện Giai đoạn 1 (Học né chướng ngại vật) ---")
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS_PHASE1,
        callback=checkpoint_callback,
        progress_bar=True
    )

    final_model_path = f"{model_dir}/uav_phase1_final.zip"
    model.save(final_model_path)
    print(f"Huấn luyện Giai đoạn 1 hoàn tất. Model được lưu tại: {final_model_path}")

    vec_env.close()
