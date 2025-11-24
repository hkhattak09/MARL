import torch
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from datetime import datetime
import gym
from gym.wrappers import AssemblySwarmWrapper
from cfg.assembly_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils import ReplayBufferAgent
from algorithm.algorithms import MADDPG


def run(cfg):
    """
    Main training function for MADDPG.
    
    Args:
        cfg: Configuration object containing training hyperparameters and settings
    """

    ## ======================================= Setup Logging =======================================
    model_dir = Path('./models') / cfg.env_name
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(str(log_dir))

    # Setup live rendering
    live_fig, live_ax = None, None
    frame_count = 0  # Counter for frame skipping in live render
    if cfg.live_render:
        plt.ion()  # Turn on interactive mode
        live_fig, live_ax = plt.subplots(figsize=(8, 8))
        live_ax.set_aspect('equal')
        live_ax.set_title('Live Training Render')
        print("Live rendering enabled - Display window will update during training")

    ## ======================================= Initialize Environment =======================================
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads)
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    elif cfg.device == 'mps':
        if torch.backends.mps.is_available():
            torch.manual_seed(cfg.seed)
        else:
            print("MPS not available, falling back to CPU.")
            cfg.device = 'cpu'

    scenario_name = 'AssemblySwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AssemblySwarmWrapper(base_env, args)
    start_stop_num = [slice(0, env.num_agents)]

    adversary_alg = None
    maddpg = MADDPG.init_from_env(
        env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg,
        tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic,
        hidden_dim=cfg.hidden_dim, device=cfg.device,
        epsilon=cfg.epsilon, noise=cfg.noise_scale, name=cfg.env_name
    )

    agent_buffer = [ReplayBufferAgent(cfg.buffer_length, env.num_agents,
                                      state_dim=env.observation_space.shape[0],
                                      action_dim=env.action_space.shape[0],
                                      start_stop_index=start_stop_num[0])]

    print("Training Starts...")
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0

        obs = env.reset()
        
        start_stop_num = [slice(0, env.n_a)]
        maddpg.prep_rollouts(device='cpu')
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            # Live rendering - update display every N frames
            if cfg.live_render and frame_count % 10 == 0:
                frame = env.render(mode='rgb_array')
                live_ax.clear()
                live_ax.imshow(frame)
                live_ax.set_title(f'Episode {ep_index}/{cfg.n_episodes} - Step {et_index}/{cfg.episode_length}')
                live_ax.axis('off')
                plt.pause(0.001)  # Brief pause to update display
            
            frame_count += 1

            torch_obs = torch.Tensor(obs).requires_grad_(False)
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=True)
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            next_obs, rewards, dones, _, agent_actions_prior = env.step(agent_actions)
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones,
                                 start_stop_num[0], agent_actions_prior)
            obs = next_obs

            episode_reward_mean_bar += np.mean(rewards)
            episode_reward_std_bar += np.std(rewards)

        end_time_1 = time.time()
        
        start_time_2 = time.time()

        # Training phase - maximize GPU utilization with larger batches
        maddpg.prep_training(device=cfg.device)
        # Increase training iterations for better sample efficiency with larger batches
        num_updates = 30 if cfg.batch_size >= 2048 else 20
        for _ in range(num_updates):
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(
                        cfg.batch_size,
                        to_gpu=True if cfg.device == 'gpu' else False,
                        is_prior=True if cfg.training_method == 'llm_rl' else False
                    )
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample,
                                  dones_sample, a_i, acs_prior_sample, env.alpha, logger=logger)

            maddpg.update_all_targets()

        maddpg.prep_rollouts(device='cpu')
        maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        env.env.alpha = 0.1
        end_time_2 = time.time()

        if ep_index % 10 == 0:
            print(f"Episodes {ep_index} of {cfg.n_episodes}, agent num: {env.n_a}, "
                  f"episode reward: {episode_reward_mean_bar/cfg.episode_length:.3f}, "
                  f"step time: {end_time_1 - start_time_1:.3f}, "
                  f"training time: {end_time_2 - start_time_2:.3f}")

        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0
            logger.add_scalars('agent/data', {
                'episode_reward_mean_bar': episode_reward_mean_bar/cfg.episode_length,
                'episode_reward_std_bar': episode_reward_std_bar/cfg.episode_length,
                'ALIGN_epi': ALIGN_epi
            }, ep_index)

        if ep_index % (4*cfg.save_interval) < cfg.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / f'model_ep{ep_index+1}.pt')

    # Save final model
    maddpg.prep_training(device=cfg.device)
    maddpg.save(run_dir / 'model.pt')
    
    # Close live rendering
    if cfg.live_render and live_fig is not None:
        plt.ioff()  # Turn off interactive mode
        print("Live rendering window closed")

    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    plt.close('all')
    
    print(f"\nTraining completed! Models saved to: {run_dir}")


if __name__ == '__main__':
    """Entry point for training script."""
    run(args)