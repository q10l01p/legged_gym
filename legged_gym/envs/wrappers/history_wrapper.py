import isaacgym
assert isaacgym
import torch
import gym


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.num_acts_history = self.obs_history_length * self.num_actions
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

        self.acts_history = torch.zeros(self.env.num_envs, self.num_acts_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        self.acts_history = torch.cat((self.acts_history[:, self.env.num_actions:], action), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history,
                'acts_history': self.acts_history}, rew, done, info

    def get_observations(self):
        """
        获取环境的当前观测、特权观测和观测历史

        Returns:
        - 一个字典，包括：
           - 'obs'：当前普通观测
           - 'privileged_obs'：当前特权观测（例如，可能含有额外信息的观测）
           - 'obs_history'：历史观测的累积

        注解:
        1. 从环境中获取当前普通观测。
        2. 从环境中获取特权观测，这种观测可能包括额外信息。
        3. 更新观测历史，将新的观测和过去的观测连接起来。
        4. 返回包含普通观测、特权观测和观测历史的字典。

        注意：函数假设self.env有get_observations和get_privileged_observations方法。
        """
        obs = self.env.get_observations()  # 获取环境的当前普通观测
        privileged_obs = self.env.get_privileged_observations()  # 获取环境的当前特权观测

        # 返回包含当前观测、特权观测以及观测历史的字典
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history,
                'acts_history': self.acts_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history,
                "acts_history": self.acts_history}


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from go1_gym_learn.ppo import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo.actor_critic import AC_Args

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()
