import ray  # .. of course
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
from scipy.stats import sem  # standard error of the mean

from recommender_system_env import RecommSys001

env = RecommSys001(config={
    "num_features": 20,  # E (embedding size)

    "num_items_in_db": 100,  # total number of items in our database
    "num_items_to_select_from": 10,  # number of items to present to the agent to pick a k-slate from
    "slate_size": 1,  # k
    "num_users_in_db": 1,  # total number  of users in our database
})
env


# !LIVE CODING!

def env_test(env):

    # 1) Reset the env.
    obs = env.reset()

    # Number of episodes already done.
    num_episodes = 0
    # Current episode's accumulated reward.
    episode_reward = 0.0
    # Collect all episode rewards here to be able to calculate a random baseline reward.
    episode_rewards = []

    # 2) Enter an infinite while loop (to step through the episode).
    while num_episodes < 1000:
        # 3) Calculate agent's action, using random sampling via the environment's action space.
        action = env.action_space.sample()
        # action = trainer.compute_single_action([obs])

        # 4) Send the action to the env's `step()` method to receive: obs, reward, done, and info.
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        # 5) Check, whether the episde is done, if yes, break out of the while loop.
        if done:
            #print(f"Episode done - accumulated reward={episode_reward}")
            num_episodes += 1
            env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    # 6) Print out mean episode reward!
    env_mean_random_reward = np.mean(episode_rewards)
    print(f"Mean episode reward when acting randomly: {env_mean_random_reward:.2f}+/-{sem(episode_rewards):.2f}")

    return env_mean_random_reward, sem(episode_rewards)



if __name__ == "__main__":
    env_mean_random_reward, env_sem_random_reward = env_test(env)

    # Start a new instance of Ray (when running this tutorial locally) or
    # connect to an already running one (when running this tutorial through Anyscale).

    ray.init(address="auto") # Hear the engine humming? ;)

    # In case you encounter the following error during our tutorial: `RuntimeError: Maybe you called ray.init twice by accident?`
    # Try: `ray.shutdown() + ray.init()` or `ray.init(ignore_reinit_error=True)`


    # Import a Trainable (one of RLlib's built-in algorithms):
    # We start our endeavor with the Bandit algorithms here b/c they are specialized in solving
    # n-arm/recommendation problems.
    from ray.rllib.agents.bandit import BanditLinUCBTrainer

    # Environment wrapping tools for:
    # a) Converting MultiDiscrete action space (k-slate recommendations) down to Discrete action space (we only have k=1 for now anyways).
    # b) Making sure our google RecSim-style environment is understood by RLlib's Bandit Trainers.
    from ray.rllib.env.wrappers.recsim import MultiDiscreteToDiscreteActionWrapper, \
        RecSimObservationBanditWrapper

    from ray import tune

    tune.register_env(
        "recomm-sys-001-for-bandits",
        lambda config: RecSimObservationBanditWrapper(MultiDiscreteToDiscreteActionWrapper(RecommSys001(config))))

    bandit_config = {
        # Use our tune-registered "RecommSys001" class.
        "env": "recomm-sys-001-for-bandits",
        "env_config": {
            "num_features": 20,  # E

            "num_items_in_db": 100,
            "num_items_to_select_from": 10,  # D
            "slate_size": 1,  # k=1

            "num_users_in_db": 1,
        },
        #"evaluation_duration_unit": "episodes",
        "timesteps_per_iteration": 1,
    }

    # Create the RLlib Trainer using above config.
    bandit_trainer = BanditLinUCBTrainer(config=bandit_config)

    # Train for n iterations (timesteps) and collect n-arm rewards.
    rewards = []
    for _ in range(300):
        result = bandit_trainer.train()
        rewards.append(result["episode_reward_mean"])
        print(".", end="")
    print(rewards)
    # Plot per-timestep (episode) rewards.
    plt.figure(figsize=(10,7))
    plt.plot(rewards)#x=[i for i in range(len(rewards))], y=rewards, xerr=None, yerr=[sem(rewards) for i in range(len(rewards))])
    plt.title("Mean reward")
    plt.xlabel("Time/Training steps")

    # Add mean random baseline reward (red line).
    plt.axhline(y=env_mean_random_reward, color="r", linestyle="-")
    plt.savefig("bandit_lin_ucb_reward.png")
    plt.show()



    # Update our env_config: Making things harder.
    bandit_config.update({
        "env_config": {
            "num_features": 20,  # E (no change)

            "num_items_in_db": 100,  # (no change)
            "num_items_to_select_from": 10,  # D (no change)
            "slate_size": 2,  # k=2

            "num_users_in_db": None,  # More users!
            "user_time_budget": 10.0,  # Longer episodes.
        },
    })

    # Re-computing our random baseline.
    harder_env = RecommSys001(config=bandit_config["env_config"])
    harder_env_mean_random_reward, _ = env_test(harder_env)


    # Create the RLlib Trainer using above config.
    bandit_trainer = BanditLinUCBTrainer(config=bandit_config)

    # Train for n iterations (timesteps) and collect n-arm rewards.
    rewards = []
    for _ in range(1200):
        result = bandit_trainer.train()
        rewards.append(result["episode_reward_mean"])
        print(".", end="")

    # Plot per-timestep (episode) rewards.
    plt.figure(figsize=(10,7))
    plt.plot(rewards)#x=[i for i in range(len(rewards))], y=rewards, xerr=None, yerr=[sem(rewards) for i in range(len(rewards))])
    plt.title("Mean reward")
    plt.xlabel("Time/Training steps")

    # Add mean random baseline reward (red line).
    plt.axhline(y=harder_env_mean_random_reward, color="r", linestyle="-")
    plt.savefig("bandit_lin_ucb_reward_harder.png")
    plt.show()