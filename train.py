from os import stat
from core.argparser import parse_args
from core.helpers import (compute_td_loss,
                          initialize_models,
                          is_atari,
                          set_device,
                          update_epsilon)
from core.replay_buffer import ReplayBuffer
from core.train_information import TrainInformation
from core.wrappers import wrap_environment

from torch import save
from torch.optim import Adam

import logging
import coloredlogs

import aicrowd_gym
import minerl

import torch
import torch.nn as nn
import torch.optim as optim

from config import EVAL_EPISODES, EVAL_MAX_STEPS

coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLBasaltFindCave-v0'

def update_graph(model, target_model, optimizer, replay_buffer, args, device,
                 info):
    if len(replay_buffer) > args.initial_learning:
        if not info.index % args.target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        batch = replay_buffer.sample(args.batch_size)
        compute_td_loss(model, target_model, batch, args.gamma, device)
        optimizer.step()


def complete_episode(model, environment, info, episode_reward, episode,
                     epsilon):
    new_best = info.update_rewards(episode_reward)
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        save(model.state_dict(), '%s.dat' % environment)
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s' % (episode, episode_reward, info.best_reward,
                           round(info.average, 3), round(epsilon, 4)))


def run_episode(env, model, target_model, optimizer, replay_buffer, args,
                device, info, episode):
    episode_reward = 0.0
    state = env.reset()
    img = env.render()

    for step_counter in range(EVAL_MAX_STEPS):
        epsilon = update_epsilon(info.index, args)
        action = model.act(img, device, epsilon)
        episode_reward -= 8*action['inventory']
        action['ESC'] = 0
        action['inventory'] = 0
        # action['camera'][0] = 0
        img = env.render()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += action['forward'] - action['back'] + 2*action['attack']
        episode_reward += reward*100
        info.update_index()
        update_graph(model, target_model, optimizer, replay_buffer, args,
                     device, info)
        if done:
            break
    complete_episode(model, args.environment, info, episode_reward,
                     episode, epsilon)
    # episode and reward print
    # print(f"[{episode}] Episode complete with reward {episode_reward}")


def train(env, model, target_model, optimizer, replay_buffer, args, device):
    info = TrainInformation()

    for episode in range(args.num_episodes):
        run_episode(env, model, target_model, optimizer, replay_buffer, args,
                    device, info, episode)

def main():
    args = parse_args()
    env = wrap_environment(args.environment)
    device = set_device(args.force_cpu)
    model, target_model = initialize_models(env, device, args.checkpoint)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_capacity)
    train(env, model, target_model, optimizer, replay_buffer, args, device)
    env.close()

def mainz():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    # model = None

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        for step_counter in range(EVAL_MAX_STEPS):

            # Step your model here.
            # Currently, it's doing random actions
            random_act = env.action_space.sample()

            # print action
            print("Action: ", random_act)

            obs, reward, done, info = env.step(random_act)
            img = env.render()

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()