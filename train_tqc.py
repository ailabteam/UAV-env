# train_tqc.py

import os
from datetime import datetime
import torch

# Import TQC từ thư viện contrib
from sb3_contrib import TQC

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Vẫn sử dụng môi trường v7, môi trường tốt nhất của chúng ta
from uav_env import UAVNetworkEnv 

# --- CẤU HÌNH HUẤN LUYỆN ---
TRAIN_TIMESTEPS = 2_000_000
NUM_ENVS = 8 # TQC cũng hưởng lợi từ việc thu thập dữ liệu song song
CHECKPOINT_FREQ = 50_000

if __name__ == '__main__':
    # Tạo thư mục riêng cho lần chạy TQC
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/TQC_{current_time}/"
    model_dir = f"models/TQC_{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        """Hàm tạo môi trường (sử dụng UAVNetworkEnv v7)"""
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        env = Monitor(env)
        return env

    # Sử dụng SubprocVecEnv để chạy song song trên nhiều lõi CPU
    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])

    # Callback để lưu model định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=model_dir,
        name_prefix="uav_tqc_model"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    # Khởi tạo model TQC
    # Chúng ta sẽ bắt đầu với các siêu tham số mặc định của TQC,
    # vốn đã được tinh chỉnh và thường hoạt động rất tốt.
    model = TQC(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        # TQC có buffer_size mặc định lớn (1_000_000), điều này tốt.
        # Các siêu tham số khác như learning_rate (7.3e-4) cũng đã được tối ưu.
    )

    print("--- Bắt đầu Huấn luyện với Thuật toán TQC (SAC Nâng cao) ---")
    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
        # Lưu lại model cuối cùng
        final_model_path = f"{model_dir}/uav_tqc_model_final.zip"
        model.save(final_model_path)
        print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")

    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C để dừng sớm
        print("\nPhát hiện KeyboardInterrupt. Đang lưu model hiện tại...")
        interrupted_model_path = f"{model_dir}/uav_tqc_model_interrupted.zip"
        model.save(interrupted_model_path)
        print(f"Đã lưu model tại: {interrupted_model_path}")
    finally:
        # Luôn đóng môi trường
        vec_env.close()
