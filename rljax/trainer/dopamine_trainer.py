import os
from statistics import median
from time import sleep, time
import numpy as np
import pandas as pd

from rljax.trainer.base_trainer import Trainer

class DopamineTrainer(Trainer):
    """
    The evaluation happens while the algoritmo is training
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        action_repeat=1,
        num_agent_steps=10 ** 5,
        save_params=True,
        num_iterations=10,
    ):
        super(DopamineTrainer, self).__init__(
            env=env,
            env_test=env_test,
            algo=algo,
            log_dir=log_dir,
            seed=seed,
            action_repeat=action_repeat,
            num_agent_steps=num_agent_steps,
            save_params=save_params,
        )
        self.num_iterations = num_iterations

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()
        self.algo.clear_iteration()

        for iteration in range(1, self.num_iterations + 1):
            for step in range(1, self.num_agent_steps + 1):
                state = self.algo.step(self.env, state)

                if self.algo.is_update():
                    self.algo.update(self.writer)

            self.save_logs(iteration)
            self.algo.clear_iteration()

            if self.save_params:
                self.algo.save_params(os.path.join(self.param_dir, f"step{iteration*self.num_agent_steps}"))

            # Wait for the logging to be finished.
            sleep(2)

    def save_logs(self, iteration):
        time = self.time      
       
        mean_return = sum(self.algo.returns)/len(self.algo.returns)

        # To TensorBoard.
        self.writer.add_scalar("return/test", mean_return, iteration * self.num_agent_steps * self.action_repeat)
        # To CSV.
        self.log["step"].append(iteration * self.num_agent_steps * self.action_repeat)
        self.log["time"].append(time)
        self.log["return"].append(str(self.algo.returns))
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to standard output.
        print(f"Num steps: {iteration * self.num_agent_steps * self.action_repeat:<6}   Return: {mean_return:<5.1f}   Time: {time}")
