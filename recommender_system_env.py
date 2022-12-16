

"""Examples for recommender system simulating envs ready to be used by RLlib Trainers
"""
import gym
import numpy as np
from typing import List, Optional

from ray.rllib.utils.numpy import softmax


class RecommSys001(gym.Env):

    def __init__(self, config=None):

        config = config or {}

        # E (embedding size)
        self.num_features = config["num_features"]
        # D
        self.num_items_to_select_from = config["num_items_to_select_from"]
        # k
        self.slate_size = config["slate_size"]

        self.num_items_in_db = config.get("num_items_in_db")
        self.items_db = None
        # Generate an items-DB containing n items, once.
        if self.num_items_in_db is not None:
            self.items_db = [np.random.uniform(0.0, 1.0, size=(self.num_features,))
                            for _ in range(self.num_items_in_db)]

        self.num_users_in_db = config.get("num_users_in_db")
        self.users_db = None
        # Store the user that's currently undergoing the episode/session.
        self.current_user = None

        # How much time does the user have to consume
        self.user_time_budget = config.get("user_time_budget", 1.0)
        self.current_user_budget = self.user_time_budget

        self.observation_space = gym.spaces.Dict({
            # The D items our agent sees at each timestep. It has to select a k-slate
            # out of these.
            "doc": gym.spaces.Dict({
                str(idx):
                    gym.spaces.Box(0.0, 1.0, shape=(self.num_features,), dtype=np.float32)
                    for idx in range(self.num_items_to_select_from)
            }),
            # The user engaging in this timestep/episode.
            "user": gym.spaces.Box(0.0, 1.0, shape=(self.num_features,), dtype=np.float32),
            # For each item in the previous slate, was it clicked? If yes, how
            # long was it being engaged with (e.g. watched)?
            "response": gym.spaces.Tuple([
                gym.spaces.Dict({
                    # Clicked or not?
                    "click": gym.spaces.Discrete(2),
                    # Engagement time (how many minutes watched?).
                    "watch_time": gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                }) for _ in range(self.slate_size)
            ]),
        })
        # Our action space is
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_items_to_select_from for _ in range(self.slate_size)
        ])

    def reset(self):
        # Reset the current user's time budget.
        self.current_user_budget = self.user_time_budget

        # Sample a user for the next episode/session.
        # Pick from a only-once-sampled user DB.
        if self.num_users_in_db is not None:
            if self.users_db is None:
                self.users_db = [np.random.uniform(0.0, 1.0, size=(self.num_features,))
                                 for _ in range(self.num_users_in_db)]
            self.current_user = self.users_db[np.random.choice(self.num_users_in_db)]
        # Pick from an infinite pool of users.
        else:
            self.current_user = np.random.uniform(0.0, 1, size=(self.num_features,))

        return self._get_obs()

    def step(self, action):
        # Action is the suggested slate (indices of the items in the suggested ones).

        scores = [np.dot(self.current_user, item)
                  for item in self.currently_suggested_items]
        best_reward = np.max(scores)

        # User choice model: User picks an item stochastically,
        # where probs are dot products between user- and item feature
        # vectors.
        # There is also a no-click item whose weight is 1.0.
        user_item_overlaps = np.array([scores[a] for a in action] + [1.0])
        which_clicked = np.random.choice(
            np.arange(self.slate_size + 1), p=softmax(user_item_overlaps))

        # Reward is the overlap, if clicked. 0.0 if nothing clicked.
        reward = 0.0
        # If anything clicked, deduct from the current user's time budget and compute
        # reward.
        if which_clicked < self.slate_size:
            regret = best_reward - user_item_overlaps[which_clicked]
            reward = 1.0 - regret
            self.current_user_budget -= 1.0
        done = self.current_user_budget <= 0.0

        # Compile response.
        response = tuple({
            "click": int(idx == which_clicked),
            "watch_time": reward if idx == which_clicked else 0.0,
        } for idx in range(len(user_item_overlaps) - 1))

        # Return 4-tuple: Next-observation, reward, done (True if episode has terminated), info dict (empty; not used here).
        return self._get_obs(response=response), reward, done, {}

    def _get_obs(self, response=None):
        # Sample D items from infinity or our pre-existing items.
        # Pick from a only-once-sampled items DB.
        if self.num_items_in_db is not None:
            self.currently_suggested_items = [
                self.items_db[item_idx].astype(np.float32)
                for item_idx in np.random.choice(self.num_items_in_db,
                                                size=(self.num_items_to_select_from,),
                                                replace=False)
            ]
        # Pick from an infinite pool of itemsdocs.
        else:
            self.currently_suggested_items = [
                np.random.uniform(0.0, 1, size=(self.num_features,)).astype(np.float32)
                for _ in range(self.num_items_to_select_from)
            ]

        return {
            "user": self.current_user.astype(np.float32),
            "doc": {
                str(idx): item for idx, item in enumerate(self.currently_suggested_items)
            },
            "response": response if response else self.observation_space["response"].sample()
        }