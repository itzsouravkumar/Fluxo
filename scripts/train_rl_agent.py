#!/usr/bin/env python3
"""Train FLUXO PPO agent on SUMO traffic simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("FLUXO RL Agent Training")
    print("=" * 40)
    print("This script trains the PPO agent on SUMO simulation.")
    print()
    print("Prerequisites:")
    print("  1. SUMO installed (apt install sumo sumo-tools)")
    print("  2. SUMO network built (./scripts/build_sumo_network.sh)")
    print("  3. pip install sumo-rl stable-baselines3")
    print()

    try:
        from sumo_rl import SumoEnvironment
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install sumo-rl stable-baselines3 gymnasium")
        return

    net_file = "core/rl/sumo/veerannapalya.net.xml"
    route_file = "core/rl/sumo/traffic_demand.rou.xml"

    if not Path(net_file).exists():
        print(f"SUMO network not found: {net_file}")
        print("Run ./scripts/build_sumo_network.sh first")
        return

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name="outputs/rl_training",
        use_gui=False,
        num_seconds=3600,
        min_green=15,
        max_green=120,
        delta_time=5,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    print("Training for 100,000 timesteps...")
    model.learn(total_timesteps=100_000)

    output_path = "models/fluxo_rl_agent_v1.zip"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
