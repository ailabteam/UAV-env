# train.py

import os
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from uav_env import UAVNetworkEnv

# --- CẤU HÌNH HUẤN LUYỆN ---
TRAIN_TIMESTEPS = 2_000_000
NUM_ENVS = 8
LOG_INTERVAL = 1
CHECKPOINT_FREQ = 50_000

# ======================== THAY ĐỔI SIÊU THAM SỐ ========================
# Đây là các siêu tham số được tinh chỉnh cho bài toán khó hơn
PPO_HYPERPARAMS = {
    "n_steps": 8192,  # Tăng số bước để có ước tính tốt hơn
    "batch_size": 512, # Giảm batch size một chút cho phù hợp
    "gamma": 0.99,
    "learning_rate": 1e-4, # Giảm learning rate để học ổn định hơn
    "ent_coef": 0.01, # Khuyến khích khám phá
    "clip_range": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
}
# ======================================================================

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{current_time}/"
    model_dir = f"models/{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        env = Monitor(env)
        return env

    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_ppo_model"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    # Sử dụng các siêu tham số đã được tinh chỉnh
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        **PPO_HYPERPARAMS # Truyền các siêu tham số vào đây
    )

    print("Bắt đầu huấn luyện AI Agent với các siêu tham số được tinh chỉnh...")
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=LOG_INTERVAL,
        progress_bar=True
    )

    final_model_path = f"{model_dir}/uav_ppo_model_final.zip"
    model.save(final_model_path)
    print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")

    vec_env.close()
