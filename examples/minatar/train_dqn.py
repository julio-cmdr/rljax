import argparse
import os
from datetime import datetime

from rljax.algorithm import DQN
from rljax.env import make_minatar_env
from rljax.trainer import DopamineTrainer


def run(args):
    env = make_minatar_env(args.env_id)
    env_test = make_minatar_env(args.env_id)

    algo = DQN(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        buffer_size=100000,
        start_steps=1000,
        update_interval_target=1000,
        units=(128,),
        env_type='minatar',
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

    trainer = DopamineTrainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        action_repeat=1,
        seed=args.seed,
        num_iterations=args.num_iterations,
    )
    trainer.train()


if __name__ == "__main__":
    for game in ["asterix", "breakout", "seaquest", "freeway", "space_invaders"]:
        p = argparse.ArgumentParser()
        p.add_argument("--env_id", type=str, default=game)
        p.add_argument("--num_agent_steps", type=int, default=1000000)
        p.add_argument("--num_iterations", type=int, default=20)
        p.add_argument("--seed", type=int, default=0)
        args = p.parse_args()
        run(args)
