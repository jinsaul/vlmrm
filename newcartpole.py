import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import clip
from PIL import Image

# HYPERPARAMETERS
LEARNING_RATE = 0.1  # Alpha
DISCOUNT_FACTOR = 0.99  # Gamma
# Reduced EPISODES significantly due to CLIP's computational cost.
# Adjust as needed, but be aware training will be very slow.
EPISODES = 1000 # Total episodes for training (was 20000)

# Exploration parameters
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY_RATE = 0.999 # Adjusted for fewer episodes (was 0.99995)
MIN_EPSILON = 0.01 # Minimum exploration rate

# --- NEW: DEBUG MODE ---
# Set to True to visualize each frame and its CLIP reward, pausing for user input.
DEBUG_MODE = False

# State discretization parameters
# Number of buckets for each dimension of the state space
# state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
NUM_BUCKETS = (6, 6, 12, 12)  # (pos, vel, ang, ang_vel)

# Visualization parameters
PLOT_REWARDS_EVERY_N_EPISODES = 10 # Adjusted for fewer episodes

# CLIP-related parameters
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-L/14@336px"  # Using a larger model for better performance
# Example prompts for CLIP. These are crucial and may need tuning.
POSITIVE_PROMPT = "perpendicular brown stick on black rectangle"
NEGATIVE_PROMPT_FALLING = "diagonal brown stick on black rectangle"
NEGATIVE_PROMPT_LEFT_EDGE = "shape on far left"
NEGATIVE_PROMPT_RIGHT_EDGE = "shape on far right"


def discretize_state(state, env):
    """
    Discretizes the continuous state from the CartPole environment.
    This remains unchanged as the Q-table is still based on numerical states.
    Args:
        state (tuple or np.array): The continuous state [pos, vel, ang, ang_vel].
        env (gym.Env): The CartPole environment instance.

    Returns:
        tuple: The discretized state.
    """
    upper_bounds = [env.observation_space.high[0], 3.0, env.observation_space.high[2], 3.0]
    lower_bounds = [env.observation_space.low[0], -3.0, env.observation_space.low[2], -3.0]
    
    bucket_indices = []
    for i in range(len(state)):
        if state[i] <= lower_bounds[i]:
            bucket_index = 0
        elif state[i] >= upper_bounds[i]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = upper_bounds[i] - lower_bounds[i]
            offset = (state[i] - lower_bounds[i]) / bound_width
            scaling = NUM_BUCKETS[i] -1 
            bucket_index = int(round(offset * scaling))
            bucket_index = max(0, min(bucket_index, NUM_BUCKETS[i] - 1))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)

def get_clip_reward(image_frame, clip_model, clip_preprocess, device, positive_text, negative_texts):
    """
    Calculates a reward based on the method from "Vision-Language Models are
    Zero-Shot Reward Models for Reinforcement Learning".
    Reward = log P(image | positive_prompt) - log P(image | negative_prompt_ensemble)
    This is approximated by: pos_logit - logsumexp(neg_logits)

    Args:
        image_frame (np.array): The RGB array of the current environment frame.
        clip_model: The loaded CLIP model.
        clip_preprocess: The CLIP image preprocessor.
        device (str): "cpu" or "cuda".
        positive_text (str): Text describing a desirable state.
        negative_texts (list of str): Texts describing undesirable states.

    Returns:
        tuple: (
            reward (float),
            positive_logit (float),
            logsumexp_of_negative_logits (float)
        )
    """
    try:
        pil_image = Image.fromarray(image_frame)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
        
        all_texts = [positive_text] + negative_texts
        text_inputs = clip.tokenize(all_texts).to(device)

        with torch.no_grad():
            # Get the raw logits from the model, as specified by the VLMRM paper.
            logits_per_image, _ = clip_model(image_input, text_inputs)
            logits = logits_per_image.squeeze()

            # Separate positive and negative logits
            positive_logit = logits[0]
            negative_logits = logits[1:]

            # --- VLMRM Reward Calculation ---
            logsumexp_neg_logits = torch.logsumexp(negative_logits, dim=0)
            reward = (positive_logit - logsumexp_neg_logits).item()

        return reward, positive_logit.item(), logsumexp_neg_logits.item()

    except Exception as e:
        print(f"Error in get_clip_reward: {e}")
        # Return neutral values on failure
        return 0.0, 0.0, 0.0

def train_agent():
    """
    Trains a Q-learning agent for the CartPole-v1 environment using CLIP-based rewards.
    """
    global EPSILON

    # Initialize CLIP model
    print(f"Loading CLIP model ({CLIP_MODEL_NAME}) on device: {CLIP_DEVICE}...")
    try:
        clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=CLIP_DEVICE)
        clip_model.eval()
        print("CLIP model loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}. CLIP rewards will not be functional.")
        return None, []

    env = gym.make("CartPole-v1", render_mode="rgb_array") 
    clip_negative_prompts = [NEGATIVE_PROMPT_FALLING, NEGATIVE_PROMPT_LEFT_EDGE, NEGATIVE_PROMPT_RIGHT_EDGE]

    # --- GOAL-BASELINE REGULARISATION ---
    print("Calculating goal-baseline reward...")
    # Get the initial frame of the environment as the "goal state"
    _, _ = env.reset()
    goal_frame = env.render()
    # Calculate the reward for this goal state to use as a baseline
    goal_baseline_reward, _, _ = get_clip_reward(
        goal_frame, clip_model, clip_preprocess, CLIP_DEVICE, POSITIVE_PROMPT, clip_negative_prompts
    )
    print(f"Goal-Baseline Reward calculated: {goal_baseline_reward:.4f}")
    
    q_table_shape = NUM_BUCKETS + (env.action_space.n,)
    q_table = np.zeros(q_table_shape)

    episode_rewards = []
    total_steps = 0

    print(f"Training for {EPISODES} episodes using CLIP rewards...")
    if DEBUG_MODE:
        print("\n--- DEBUG MODE ENABLED ---")
        print("A plot will show each frame. Press 'Enter' in the console to advance to the next step.")

    start_time = time.time()

    # --- Setup for Debug Mode Visualization ---
    debug_fig = None
    if DEBUG_MODE:
        plt.ion()  # Turn on interactive mode for matplotlib
        debug_fig = plt.figure(figsize=(8, 6))

    for episode in range(EPISODES):
        initial_state_continuous, info = env.reset()
        current_discrete_state = discretize_state(initial_state_continuous, env)
        
        terminated = False
        truncated = False
        episode_reward_sum = 0 

        while not terminated and not truncated:
            if np.random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_discrete_state])

            next_state_continuous, env_reward, terminated, truncated, info = env.step(action)
            
            current_frame = env.render() 
            if current_frame is None:
                print("Warning: env.render() returned None. Cannot compute CLIP reward.")
                regularized_clip_reward = env_reward # Fallback to environment reward
                clip_reward, pos_logit, logsumexp_neg = 0.0, 0.0, 0.0
            else:
                # Get reward and logit values from the function
                clip_reward, pos_logit, logsumexp_neg = get_clip_reward(
                    current_frame, clip_model, clip_preprocess, CLIP_DEVICE, 
                    POSITIVE_PROMPT, clip_negative_prompts
                )
                # Apply goal-baseline regularisation
                regularized_clip_reward = clip_reward - goal_baseline_reward
            
            episode_reward_sum += regularized_clip_reward
            total_steps +=1

            # --- DEBUG MODE VISUALIZATION LOGIC ---
            if DEBUG_MODE:
                plt.figure(debug_fig.number)
                plt.clf()
                action_text = "LEFT" if action == 0 else "RIGHT"
                
                title_text = (
                    f"Episode {episode + 1} | Action: {action_text}\n"
                    f"Pos Logit: {pos_logit:.2f} | LogSumExp(Neg): {logsumexp_neg:.2f} | Raw Reward: {clip_reward:.2f}\n"
                    f"Baseline: {goal_baseline_reward:.2f} | Final Regularized Reward = {regularized_clip_reward:.2f}"
                )
                
                plt.imshow(current_frame)
                plt.title(title_text, fontsize=10)
                plt.axis('off')
                plt.pause(0.01)
                input(">>> Press Enter in console to continue...")

            next_discrete_state = discretize_state(next_state_continuous, env)
            
            # Q-learning update rule using the regularized CLIP reward
            if not terminated and not truncated:
                max_future_q = np.max(q_table[next_discrete_state])
                current_q_val = q_table[current_discrete_state + (action,)]
                new_q = current_q_val + LEARNING_RATE * (regularized_clip_reward + DISCOUNT_FACTOR * max_future_q - current_q_val)
                q_table[current_discrete_state + (action,)] = new_q
            elif terminated: 
                current_q_val = q_table[current_discrete_state + (action,)]
                q_table[current_discrete_state + (action,)] = current_q_val + LEARNING_RATE * (regularized_clip_reward - current_q_val)

            current_discrete_state = next_discrete_state
        
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY_RATE
            EPSILON = max(MIN_EPSILON, EPSILON)

        episode_rewards.append(episode_reward_sum)

        if (episode + 1) % PLOT_REWARDS_EVERY_N_EPISODES == 0:
            avg_reward = sum(episode_rewards[-PLOT_REWARDS_EVERY_N_EPISODES:]) / PLOT_REWARDS_EVERY_N_EPISODES
            print(f"Episode: {episode + 1}, Avg Regularized Reward (last {PLOT_REWARDS_EVERY_N_EPISODES}): {avg_reward:.2f}, Epsilon: {EPSILON:.4f}, Steps: {total_steps}")
            total_steps = 0

    env.close()
    
    # --- Cleanup for Debug Mode ---
    if DEBUG_MODE:
        plt.ioff()
        plt.close(debug_fig)

    print("Training finished.")
    end_time = time.time()
    print(f"Training took: {end_time - start_time:.2f} seconds")

    # Create a new figure for the final rewards plot
    plt.figure(figsize=(12, 6))
    moving_avg_rewards = np.convolve(episode_rewards, np.ones(PLOT_REWARDS_EVERY_N_EPISODES)/PLOT_REWARDS_EVERY_N_EPISODES, mode='valid')
    plt.plot(moving_avg_rewards)
    plt.title(f"Moving Average of Episode CLIP Rewards (every {PLOT_REWARDS_EVERY_N_EPISODES} eps)")
    plt.xlabel(f"Episode Chunk (averaged over {PLOT_REWARDS_EVERY_N_EPISODES} episodes)")
    plt.ylabel("Average Regularized Reward")
    plt.grid(True)
    plt.savefig("cartpole_clip_rewards.png")
    print("Rewards plot saved as cartpole_clip_rewards.png")
    plt.close('all')
    
    return q_table, episode_rewards

def test_agent(q_table, num_test_episodes=10):
    """
    Tests the trained Q-learning agent and renders its performance.
    """
    if q_table is None:
        print("Q-table is None. Cannot test agent.")
        return

    env = gym.make("CartPole-v1", render_mode="human")
    total_rewards = []

    print(f"\nTesting agent for {num_test_episodes} episodes (using standard env rewards for test eval)...")

    for episode in range(num_test_episodes):
        initial_state_continuous, info = env.reset()
        env.render()
        current_discrete_state = discretize_state(initial_state_continuous, env)
        
        terminated = False
        truncated = False
        episode_env_reward = 0
        
        while not terminated and not truncated:
            env.render()
            action = np.argmax(q_table[current_discrete_state])
            
            next_state_continuous, reward, terminated, truncated, info = env.step(action)
            next_discrete_state = discretize_state(next_state_continuous, env)
            
            episode_env_reward += reward
            current_discrete_state = next_discrete_state
            
            time.sleep(0.02)

        total_rewards.append(episode_env_reward)
        print(f"Test Episode: {episode + 1}, Env Reward: {episode_env_reward}")
        if episode_env_reward < 500:
             print("   Agent may not have learned to balance the pole for the maximum duration.")

    env.close()
    avg_test_reward = sum(total_rewards) / num_test_episodes
    print(f"\nAverage standard environment reward over {num_test_episodes} test episodes: {avg_test_reward:.2f}")


if __name__ == "__main__":
    print("Attempting to train CartPole with Q-learning using CLIP-based rewards.")
    print("WARNING: This will be VERY SLOW due to image processing at each step.")
    print("Ensure you have 'torch', 'clip', and 'Pillow' installed ('pip install torch torchvision torchaudio clip-by-openai Pillow')")
    
    trained_q_table, rewards_history = train_agent()
    
    if trained_q_table is not None and not DEBUG_MODE:
        test_agent(trained_q_table, num_test_episodes=5)
    elif DEBUG_MODE:
        print("\nTesting phase skipped as DEBUG_MODE was on.")
    else:
        print("Training failed or was skipped. Exiting.")
