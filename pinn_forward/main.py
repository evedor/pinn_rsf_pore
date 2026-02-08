# main.py
import argparse
from solver import FrictionPINNSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Friction PINN training")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--exp", type=str, default="b698",
                        help="Experiment key, you can find it in the config.yaml")

    args = parser.parse_args()

    # 启动训练
    solver = FrictionPINNSolver(
        config_path=args.config,
        experiment=args.exp
    )
    solver.train()
    solver.plot_loss()
    solver.evaluate()