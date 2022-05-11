import gym
from gym import spaces
from gym.envs import register

from minatar import Environment

class MinAtarEnv(object):
  metadata = {"render.modes": ["human", "array"]}

  def __init__(self, game_name, seed, display_time=50, use_minimal_action_set=False, _max_episode_steps=9999999, **kwargs):
    self.game_name = game_name
    self.display_time = display_time
    
    self.game_kwargs = kwargs
    
    self.game = Environment(
      env_name=self.game_name,
      random_seed=seed,
      **self.game_kwargs
    )

    if use_minimal_action_set:
      self.action_set = self.game.minimal_action_set()
    else:
      self.action_set = list(range(self.game.num_actions()))

    self.action_space = spaces.Discrete(len(self.action_set), seed=seed)
    self.observation_space = spaces.Box(
      0.0, 1.0, shape=self.game.state_shape(), dtype=bool
    )
    self._max_episode_steps = _max_episode_steps

  def step(self, action):
    action = self.action_set[action]
    reward, done = self.game.act(action)
    return self.game.state(), reward, done, {}

  def reset(self):
    self.game.reset()
    return self.game.state()

  def seed(self, seed=None):
    self.action_space = spaces.Discrete(len(self.action_set), seed=seed)
    self.game = Environment(
      env_name=self.game_name,
      random_seed=seed,
      **self.game_kwargs
    )
    return seed

  def render(self, mode="human"):
    if mode == "array":
      return self.game.state()
    elif mode == "human":
      self.game.display_state(self.display_time)

  def close(self):
    if self.game.visualized:
      self.game.close_display()
    return 0

def make_minatar_env(game_name, seed=None):
  return MinAtarEnv(game_name, seed)