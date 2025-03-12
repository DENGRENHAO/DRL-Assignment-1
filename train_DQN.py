import random
import numpy as np
from collections import deque
from simple_custom_taxi_env import SimpleTaxiEnv
from student_agent import get_action, update_q_network, soft_update_target_network, save_q_network

# Training parameters
NUM_EPISODES = 20000
MAX_STEPS = 2000
EPSILON_DECAY = 0.99975
# MIN_EPSILON = 0.1
MIN_EPSILON = 0.05
# EPSILON = 1.0
EPSILON = 0.143
BATCH_SIZE = 128

# Experience replay buffer
REPLAY_BUFFER = deque(maxlen=10000)

def train_agent():
    global EPSILON
    total_rewards = []
    
    for episode in range(16000, NUM_EPISODES):
        grid_size = random.randint(5, 10)
        # grid_size = 5
        env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=5000)
        obs, _ = env.reset()
        total_reward = 0
        episode_steps = []
        episode_step = 0
        
        for _ in range(MAX_STEPS):
            action = get_action(obs, EPSILON)
            new_obs, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            REPLAY_BUFFER.append((obs, action, reward, new_obs, done))
            
            # Update the network using experience replay
            if len(REPLAY_BUFFER) > BATCH_SIZE:
                batch = random.sample(REPLAY_BUFFER, BATCH_SIZE)
                update_q_network(batch)
                
            soft_update_target_network()
            
            obs = new_obs
            total_reward += reward
            episode_step += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
        episode_steps.append(episode_step)
        
        # Decay epsilon
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        
        # Logging progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            print(f"Episode {episode+1}/{NUM_EPISODES}, Avg Reward: {avg_reward:.2f}, Epsilon: {EPSILON:.3f}, Avg Steps: {avg_steps:.2f}")
        
        if (episode + 1) % 1000 == 0:
            save_q_network(episode + 1)
    
    # Save trained network after training
    # save_q_network()
    print("Training complete. Q-network saved.")

if __name__ == "__main__":
    train_agent()