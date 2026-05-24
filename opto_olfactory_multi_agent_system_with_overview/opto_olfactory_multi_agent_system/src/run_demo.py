import os
import pprint

from .config import SystemConfig
from .data_simulator import SyntheticOlfactoryDataGenerator
from .agents import MultiAgentOlfactorySystem
from .utils import seed_everything, save_json


def main():
    cfg = SystemConfig()
    seed_everything(cfg.seed)

    generator = SyntheticOlfactoryDataGenerator(cfg)

    # 演示：生成一个 hf_like 样本，也可以改成 methanol/ammonia/mixed_organic/safe
    target_label_name = "hf_like"
    target_label_id = list(cfg.class_names).index(target_label_name)

    time_signal, spectral_image, env, label_id = generator.generate_one(label_id=target_label_id)

    if os.path.exists(cfg.checkpoint_path):
        model_path = cfg.checkpoint_path
        print(f"Loaded trained model: {model_path}")
    else:
        model_path = None
        print("Warning: checkpoint not found. Using an untrained model. Please run: python -m src.train")

    system = MultiAgentOlfactorySystem(cfg, model_path=model_path)
    result = system.analyze(time_signal, spectral_image, env)

    result["demo_ground_truth"] = {
        "label_id": int(label_id),
        "label_name": cfg.class_names[label_id],
        "temperature_norm": float(env[0]),
        "humidity_norm": float(env[1]),
    }

    save_json(result, cfg.report_path)

    print("\n===== Safety Report =====")
    pprint.pp(result["safety_report"], width=120)
    print(f"\nReport saved to: {cfg.report_path}")


if __name__ == "__main__":
    main()
