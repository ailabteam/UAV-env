# train_tqc.py
import os
from datetime import datetime
import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from uav_env import UAVNetworkEnv 

TRAIN_TIMESTEPS = 2_000_000; NUM_ENVS = 8; CHECKPOINT_FREQ = 100_000

if __name__ == '__main__':
    run_name = f"TQC_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir, model_dir = f"logs/{run_name}/", f"models/{run_name}/"
    os.makedirs(log_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)
    def make_env():
        env = UAVNetworkEnv(num_uavs=2, num_iot_nodes=15)
        return Monitor(env)
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    checkpoint_callback = CheckpointCallback(save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1), save_path=model_dir, name_prefix="tqc_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    model = TQC("MlpPolicy", vec_env, device=device, verbose=1, tensorboard_log=log_dir)
    print(f"--- Starting TQC Training on Env v7 ---")
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)
    model.save(f"{model_dir}/tqc_model_final.zip")
    print("--- TQC Training Complete ---")
    vec_env.close()
