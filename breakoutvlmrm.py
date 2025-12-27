import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from PIL import Image
import os
from collections import deque
import random
import ale_py

# --- SCRIPT CONFIGURATION ---
# Set to True to run the training process. Set to False to skip training and only test a saved model.
TRAIN_AGENT = True
# If TRAIN_AGENT is True, set this to True to load an existing checkpoint and continue training.
LOAD_CHECKPOINT_IF_EXISTS = True
# File path for saving and loading the training state checkpoint.
CHECKPOINT_PATH = "breakout_dqn_vlmrm_checkpoint.pth"

# --- HYPERPARAMETERS ---
# DQN & Training Hyperparameters
BUFFER_SIZE = 10000          # Replay buffer size (reduced for lower memory, 100k is common)
BATCH_SIZE = 32              # Number of experiences to sample from memory
GAMMA = 0.99                 # Discount factor
EPS_START = 1.0              # Starting value of epsilon
EPS_END = 0.1                # Minimum value of epsilon
EPS_DECAY = 30000            # How fast to decay epsilon
TAU = 0.005                  # For soft update of target network
LR = 1e-4                    # Learning rate for the optimizer
UPDATE_EVERY = 4             # How often to update the network

# Training session parameters
NUM_EPISODES = 2000          # Total number of episodes to train for

# --- DEBUG MODE ---
# Set to True to visualize each frame and its CLIP reward.
DEBUG_MODE = True
# If DEBUG_MODE is True, set this to True to visualize without pausing for user input.
DEBUG_NO_PAUSE = True

LOAD_CHECKPOINT_IF_EXISTS = False

# CLIP-related parameters
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-L/14@336px"
# Prompts specifically for Breakout
POSITIVE_PROMPT = "empty"
NEGATIVE_PROMPT_LOSS = "full rainbow"

# --- Environment Wrappers for Preprocessing ---

class PreprocessFrame(gym.ObservationWrapper):
    """
    Grayscales and resizes the observation frame.
    """
    def __init__(self, env, new_size=(84, 84)):
        super(PreprocessFrame, self).__init__(env)
        self.new_size = new_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.new_size, dtype=np.uint8)

    def observation(self, obs):
        # From RGB to Grayscale
        obs = np.array(Image.fromarray(obs).convert('L'))
        # Resize
        obs = np.array(Image.fromarray(obs).resize(self.new_size, Image.NEAREST))
        return obs

class StackFrames(gym.ObservationWrapper):
    """
    Stacks the last n_frames observations.
    """
    def __init__(self, env, n_frames=4):
        super(StackFrames, self).__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        assert len(self.frames) == self.n_frames
        return np.stack(self.frames, axis=0)

# --- DQN Architecture ---

class DQN(nn.Module):
    """Deep Q-Network for Atari games."""
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Input x is (batch_size, 4, 84, 84), need to be float and normalized
        x = x.float() / 255.0
        x = self.cnn(x)
        x = self.fc(x)
        return x

# --- Replay Buffer ---

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None], axis=0)).to(CLIP_DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(CLIP_DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(CLIP_DEVICE)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None], axis=0)).to(CLIP_DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(CLIP_DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- Agent Class ---

class DQNAgent():
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.policy_net = DQN(n_actions).to(CLIP_DEVICE)
        self.target_net = DQN(n_actions).to(CLIP_DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).unsqueeze(0).to(CLIP_DEVICE)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from policy model
        Q_expected = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# --- VLMRM Reward Function ---

def get_clip_reward(image_frame, clip_model, clip_preprocess, device, positive_text, negative_texts):
    try:
        pil_image = Image.fromarray(image_frame)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        all_texts = [positive_text] + negative_texts
        text_inputs = clip.tokenize(all_texts).to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(image_input, text_inputs)
            logits = logits_per_image.squeeze()
            positive_logit = logits[0]
            negative_logits = logits[1:]
            logsumexp_neg_logits = torch.logsumexp(negative_logits, dim=0)
            reward = (positive_logit - logsumexp_neg_logits).item()
        return reward
    except Exception as e:
        print(f"Error in get_clip_reward: {e}")
        return 0.0

# --- Main Training Loop ---

def train():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = PreprocessFrame(env)
    env = StackFrames(env)
    
    agent = DQNAgent(n_actions=env.action_space.n)
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=CLIP_DEVICE)
    clip_model.eval()

    # #goalbaseline regularisation
    # goal_baseline_reward, _, _ = get_clip_reward(
    #     goal_frame, clip_model, clip_preprocess, CLIP_DEVICE, POSITIVE_PROMPT, clip_negative_prompts
    # )

    start_episode = 0
    scores_history = []
    
    # Load checkpoint if exists
    if LOAD_CHECKPOINT_IF_EXISTS and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        scores_history = checkpoint['scores_history']
        print(f"Resuming from episode {start_episode}")

    scores = []
    scores_window = deque(maxlen=100)
    eps = EPS_START

    # --- Setup for Debug Mode Visualization ---
    debug_fig = None
    if DEBUG_MODE:
        plt.ion()
        debug_fig = plt.figure(figsize=(8, 6))
        print("\n--- DEBUG MODE ENABLED ---")
        if not DEBUG_NO_PAUSE:
            print("A plot will show each frame. Press 'Enter' in the console to advance to the next step.")

    for i_episode in range(start_episode + 1, NUM_EPISODES + 1):
        state, _ = env.reset()
        score = 0
        lives = 5 # Initial lives in breakout
        
        while True:
            # Get raw frame for CLIP before action
            raw_frame = env.render()
            
            action = agent.act(state, eps)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            
            # --- VLMRM Reward Calculation ---
            clip_reward = get_clip_reward(raw_frame, clip_model, clip_preprocess, CLIP_DEVICE, 
                                          POSITIVE_PROMPT, [NEGATIVE_PROMPT_LOSS])
            
            # Check if a life was lost
            life_lost_penalty = 0
            if info['lives'] < lives:
                life_lost_penalty = -5 # Large penalty for losing a life
                lives = info['lives']
            
            final_clip_reward = clip_reward + life_lost_penalty

            # --- DEBUG MODE VISUALIZATION LOGIC ---
            if DEBUG_MODE:
                plt.figure(debug_fig.number)
                plt.clf()
                action_map = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}
                action_text = action_map.get(action, 'UNKNOWN')

                title_text = (
                    f"Episode {i_episode} | Action: {action_text}\n"
                    f"VLMRM Reward: {clip_reward:.2f} | Life Lost Penalty: {life_lost_penalty}\n"
                    f"--> Final Reward for this step = {final_clip_reward:.2f}"
                )
                
                plt.imshow(raw_frame)
                plt.title(title_text, fontsize=10)
                plt.axis('off')
                plt.pause(0.01)
                
                if not DEBUG_NO_PAUSE:
                    input(">>> Press Enter in console to continue...")

            done = terminated or truncated
            
            agent.step(state, action, final_clip_reward, next_state, done)
            state = next_state
            score += env_reward # We track the game score for evaluation
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        scores_history.append(score)
        
        eps = max(EPS_END, EPS_DECAY / (EPS_DECAY + i_episode))

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            # Save checkpoint
            torch.save({
                'episode': i_episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'scores_history': scores_history,
            }, CHECKPOINT_PATH)
            print("Checkpoint saved.")

    # --- Cleanup for Debug Mode ---
    if DEBUG_MODE:
        plt.ioff()
        plt.close(debug_fig)

    # Final Save
    torch.save({
            'episode': NUM_EPISODES,
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'scores_history': scores_history,
        }, CHECKPOINT_PATH)
    print("Final checkpoint saved.")
    
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(scores_history)), scores_history)
    plt.ylabel('Game Score')
    plt.xlabel('Episode #')
    plt.title('Training Performance (Game Score)')
    plt.savefig('breakout_training_plot.png')
    plt.show()

def test():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = PreprocessFrame(env)
    env = StackFrames(env)
    
    agent = DQNAgent(n_actions=env.action_space.n)
    
    if os.path.exists(CHECKPOINT_PATH) and LOAD_CHECKPOINT_IF_EXISTS:
        print(f"Loading model from {CHECKPOINT_PATH} for testing.")
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.policy_net.eval()
    else:
        print("No checkpoint found. Cannot test.")
        return

    for i in range(5): # Test for 5 episodes
        state, _ = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps=0.01) # Act almost deterministically
            next_state, reward, terminated, truncated, info = env.step(action)
            score += reward
            state = next_state
            if terminated or truncated:
                print(f"Test Episode {i+1} Finished. Score: {score}")
                break
    env.close()

if __name__ == "__main__":
    if TRAIN_AGENT:
        print("--- Starting Training ---")
        train()
    else:
        print("--- Starting Testing ---")
        test()
