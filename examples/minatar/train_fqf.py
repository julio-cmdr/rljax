import argparse
import os
os.environ['TF_CUDNN_DETERMINISTIC']='1'
from datetime import datetime

from rljax.algorithm import FQF
from rljax.env import make_minatar_env
from rljax.trainer import DopamineTrainer


def run(args):
    env = make_minatar_env(args.env_id, args.seed)
    env_test = make_minatar_env(args.env_id, args.seed)

    algo = FQF(
        num_agent_steps=args.num_agent_steps*args.num_iterations,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        buffer_size=100000,
        start_steps=1000,
        update_interval_target=1000,
        units=(128,),
        env_type='minatar',
        lr=2.5e-04,
        batch_size=16,
        lr_cum_p=2.5e-10,
        nstep=3,
        use_per=True,
        munchausen=True,
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
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="space_invaders")
    p.add_argument("--num_agent_steps", type=int, default=1000000)
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
