# train.py

import os
import re
from datetime import datetime
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from uav_env import UAVNetworkEnv 

# --- CẤU HÌNH ---
TRAIN_TIMESTEPS = 2_000_000
NUM_ENVS = 8
SAC_HYPERPARAMS = {
    "learning_rate": 3e-4, "buffer_size": 300_000, "batch_size": 256,
    "gamma": 0.99, "tau": 0.005, "train_freq": (1, "step"), "gradient_steps": 1,
}

# ======================== ĐÃ CẬP NHẬT SẴN ========================
# Đặt đường dẫn đến model bạn muốn tiếp tục huấn luyện.
# Nếu để là None, script sẽ bắt đầu một lần chạy mới.
RESUME_FROM_MODEL = "models/SAC_20250807_052413/uav_sac_model_1750000_steps.zip" 
# ================================================================

if __name__ == '__main__':
    if RESUME_FROM_MODEL and os.path.exists(RESUME_FROM_MODEL):
        # Nếu tiếp tục, hãy sử dụng lại thư mục log và model cũ
        try:
            run_name = RESUME_FROM_MODEL.split('/')[-2]
            log_dir = f"logs/{run_name}/"
            model_dir = f"models/{run_name}/"
            print(f"--- Tiếp tục huấn luyện từ model: {RESUME_FROM_MODEL} ---")
            print(f"--- Log sẽ được ghi tiếp vào: {log_dir} ---")
        except IndexError:
            print("Lỗi: Đường dẫn model không hợp lệ. Vui lòng kiểm tra lại.")
            exit()
    else:
        # Nếu bắt đầu mới, tạo thư mục mới
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"SAC_{current_time}"
        log_dir = f"logs/{run_name}/"
        model_dir = f"models/{run_name}/"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        print("--- Bắt đầu một lần huấn luyện mới ---")

    def make_env():
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        env = Monitor(env)
        return env

    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_sac_model"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if RESUME_FROM_MODEL and os.path.exists(RESUME_FROM_MODEL):
        # Tải model đã có và đặt lại môi trường cho nó
        model = SAC.load(RESUME_FROM_MODEL, env=vec_env, device=device)
        
        # Tự động trích xuất số bước đã huấn luyện từ tên file
        try:
            # Dùng regex để tìm số trong tên file, đáng tin cậy hơn split
            match = re.search(r'_(\d+)_steps', RESUME_FROM_MODEL)
            if match:
                timesteps_already_trained = int(match.group(1))
            else:
                timesteps_already_trained = 0
            print(f"Model đã được huấn luyện {timesteps_already_trained} timesteps.")
        except:
            timesteps_already_trained = 0
        
        timesteps_remaining = TRAIN_TIMESTEPS - timesteps_already_trained
        
    else:
        # Tạo model mới từ đầu
        model = SAC(
            policy="MlpPolicy", env=vec_env, device=device,
            verbose=1, tensorboard_log=log_dir, **SAC_HYPERPARAMS
        )
        timesteps_remaining = TRAIN_TIMESTEPS

    if timesteps_remaining > 0:
        print(f"Số timesteps còn lại cần huấn luyện: {timesteps_remaining}")
        model.learn(
            total_timesteps=timesteps_remaining,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False # Rất quan trọng: không reset bộ đếm timestep
        )

        final_model_path = f"{model_dir}/uav_sac_model_final_{TRAIN_TIMESTEPS}.zip"
        model.save(final_model_path)
        print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")
    else:
        print("Model đã được huấn luyện đủ số timesteps. Không cần huấn luyện thêm.")

    vec_env.close()
