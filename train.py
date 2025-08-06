# train.py

import os
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Import môi trường tùy chỉnh của chúng ta
from uav_env import UAVNetworkEnv

# --- CẤU HÌNH HUẤN LUYỆN ---
# Các tham số này rất quan trọng để có kết quả tốt
# Bạn sẽ cần tinh chỉnh chúng trong quá trình nghiên cứu
TRAIN_TIMESTEPS = 2_000_000   # Tổng số bước huấn luyện
NUM_ENVS = 8                 # Số môi trường chạy song song (tận dụng CPU đa lõi)
LOG_INTERVAL = 1             # Tần suất in log (mỗi N lần cập nhật)
CHECKPOINT_FREQ = 50_000     # Tần suất lưu lại model (mỗi N bước)

if __name__ == '__main__':
    # 1. Thiết lập thư mục lưu trữ
    # Tạo một thư mục log và model riêng cho mỗi lần chạy dựa trên thời gian
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{current_time}/"
    model_dir = f"models/{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Tạo các môi trường song song (Vectorized Environment)
    # Đây là kỹ thuật cực kỳ quan trọng để tăng tốc độ thu thập dữ liệu
    # Nó tạo ra NUM_ENVS môi trường chạy trên các tiến trình CPU khác nhau
    def make_env():
        return UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)

    # Nếu bạn dùng Windows, có thể cần SubprocVecEnv. Nếu Linux/Mac, DummyVecEnv cũng ổn
    # SubprocVecEnv thường nhanh hơn trên hệ thống đa lõi
    vec_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])
    # vec_env = DummyVecEnv([make_env for i in range(NUM_ENVS)])


    # 3. Tạo Callback để lưu model định kỳ
    # Rất quan trọng để không bị mất tiến trình huấn luyện nếu có sự cố
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1), # Điều chỉnh freq cho vec_env
        save_path=model_dir,
        name_prefix="uav_ppo_model"
    )

    # 4. Định nghĩa AI Agent (PPO)
    # Đây là lúc chúng ta bảo PyTorch sử dụng GPU!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device.upper()}")
    
    # Các tham số của PPO có thể được tinh chỉnh để có hiệu năng tốt hơn
    # Ví dụ: learning_rate, n_steps, batch_size, ...
    model = PPO(
        policy="MlpPolicy",      # MlpPolicy: mạng nơ-ron tiêu chuẩn
        env=vec_env,             # Huấn luyện trên các môi trường song song
        device=device,           # Yêu cầu sử dụng GPU
        verbose=1,               # In ra thông tin huấn luyện
        tensorboard_log=log_dir  # Lưu log để xem bằng TensorBoard
    )

    # 5. Bắt đầu quá trình huấn luyện
    print("Bắt đầu huấn luyện AI Agent...")
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=LOG_INTERVAL,
        progress_bar=True # Hiển thị thanh tiến trình đẹp mắt
    )

    # 6. Lưu lại model cuối cùng sau khi huấn luyện xong
    final_model_path = f"{model_dir}/uav_ppo_model_final.zip"
    model.save(final_model_path)
    print(f"Huấn luyện hoàn tất. Model cuối cùng được lưu tại: {final_model_path}")

    # Đóng các môi trường
    vec_env.close()
