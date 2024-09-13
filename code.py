
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class PopSpikeEncoderRegularSpike(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.mean_range = torch.linspace(-mean_range, mean_range, pop_dim).to(device)
        self.std = std

    def forward(self, obs, batch_size):
        obs_expanded = obs.unsqueeze(-1).repeat(1, 1, self.pop_dim)  # Shape: [batch_size, obs_dim, pop_dim]
        mean_range_expanded = self.mean_range.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.obs_dim, 1)  # Shape: [batch_size, obs_dim, pop_dim]
        gaussian = torch.exp(-(obs_expanded - mean_range_expanded)**2 / (2 * self.std**2))  # Shape: [batch_size, obs_dim, pop_dim]
        spikes = torch.rand(batch_size, self.obs_dim, self.pop_dim).to(self.device) < gaussian
        spikes = spikes.permute(0, 2, 1).unsqueeze(-1).repeat(1, 1, 1, self.spike_ts)  # Shape: [batch_size, pop_dim, obs_dim, spike_ts]
        return spikes.float()

class SpikeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, spike_ts, device):
        super().__init__()
        self.spike_ts = spike_ts
        self.device = device

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, batch_size):
        # Reshape the input to match the expected input dimension of the first linear layer
        # x = x.view(batch_size, -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class PopSpikeDecoder(nn.Module):
    def __init__(self, act_dim, pop_dim):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim

    def forward(self, spikes):
        return spikes.mean(dim=-1)

class SpikingActor(nn.Module):
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, device):
        super().__init__()
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim * en_pop_dim * spike_ts, act_dim * de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, batch_size):
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes.view(batch_size, -1), batch_size)
        mu = self.decoder(out_pop_activity.view(batch_size, -1, self.decoder.pop_dim))
        # Add a softmax layer to ensure the output is a probability distribution over actions
        return F.softmax(mu, dim=-1), None  # Return the softmax output


class Channel:
    def __init__(
            self,
            N_D2D=150,
            N_CU=30,
            D2D_dis=30,
            SimulationRegion=1000,
            AWGN=-174,
            W=10*10**6,
            PLfactor=4,
            PL_k=10**-2,
            CU_tr_Power=22,
            CU_min_SINR=6,
            D2D_tr_Power_levels=30,
            D2D_tr_Power_max=23,
            D2D_min_SINR=6,
    ):
        self.N_D2D = N_D2D
        self.N_CU = N_CU
        self.D2D_dis = D2D_dis
        self.SimulationRegion = SimulationRegion
        self.AWGN = AWGN
        self.W = W
        self.PLfactor = PLfactor
        self.PL_k = PL_k
        self.CU_tr_Power = CU_tr_Power
        self.CU_min_SINR = CU_min_SINR
        self.D2D_tr_Power_levels = D2D_tr_Power_levels
        self.D2D_tr_Power_max = D2D_tr_Power_max
        self.D2D_min_SINR = D2D_min_SINR

        self.action_space = np.array(range(0, self.D2D_tr_Power_levels * self.N_CU))
        self.n_actions = len(self.action_space)
        self.observation_space = self.N_CU

    def reset(self):
        # Initialize channel state
        self.g_iB = np.random.exponential(1, size=self.N_CU)
        self.g_j = np.random.exponential(1, size=self.N_D2D)
        self.G_ij = np.random.exponential(1, size=(self.N_D2D, self.N_CU))
        self.g_jB = np.random.exponential(1, size=self.N_D2D)
        self.G_j_j = np.random.exponential(1, size=(self.N_D2D, self.N_D2D))
        self.d_ij = np.random.uniform(0, self.SimulationRegion, size=(self.N_D2D, self.N_CU))

        return np.zeros(self.N_CU)  # Initial state

    def step(self, action):
        # Decode the action to get power levels for each D2D pair
        D2D_power_levels = np.array(action) // self.N_CU
        D2D_power = 10 ** ((D2D_power_levels * self.D2D_tr_Power_max / (self.D2D_tr_Power_levels - 1)) / 10)

        # Calculate SINR for each CU and D2D pair
        CU_SINR = np.zeros(self.N_CU)
        D2D_SINR = np.zeros(self.N_D2D)
        for i in range(self.N_CU):
            signal_power = self.CU_tr_Power * self.g_iB[i]
            interference_from_D2D = np.sum(D2D_power * self.G_ij[:, i] * self.d_ij[:, i] ** (-self.PLfactor))
            noise_power = 10 ** (self.AWGN / 10)
            CU_SINR[i] = signal_power / (interference_from_D2D + noise_power)

        for j in range(self.N_D2D):
            signal_power = D2D_power * self.g_j[j]
            interference_from_CU = np.sum(self.CU_tr_Power * self.G_ij[j, :] * self.d_ij[j, :] ** (-self.PLfactor))
            interference_from_D2D = np.sum(D2D_power * self.G_j_j[j, :] * self.D2D_dis ** (-self.PLfactor)) - signal_power  # Exclude self-interference
            D2D_SINR[j] = signal_power / (interference_from_CU + interference_from_D2D + noise_power)

        # Calculate reward
        common_reward = np.sum(np.log2(1 + np.maximum(CU_SINR, 0)))  # Ensure positive input to log2
        individual_rewards = np.log2(1 + np.maximum(D2D_SINR, 0))   # Ensure positive input to log2
        reward = common_reward + np.sum(individual_rewards)

        # Termination condition
        done = np.any(CU_SINR < self.CU_min_SINR)  or np.any(D2D_SINR < self.D2D_min_SINR)

        # Generate next state (example with multiple options)
        # Option 1: Channel gains, distances, and interference levels
        #next_state = np.concatenate([
            #np.log10(self.g_iB),
            #np.log10(self.g_j),
            #np.log10(self.d_ij.flatten()),
            #np.log10(interference_from_D2D),
            #np.log10(interference_from_CU)
           #])

        # Option 2: Joint SINR of all CUs
        # next_state = np.log10(CU_SINR)

        # Option 3: Joint SINR of all CUs and D2D pairs (flattened)
        next_state = np.concatenate([np.log10(CU_SINR), np.log10(D2D_SINR)])

        info = {}
        return next_state, reward, done, info

# RL Agent
class Agent:
    def __init__(self, obs_dim, act_dim, device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        en_pop_dim = 10
        de_pop_dim = 10
        hidden_sizes = [64, 64]
        mean_range = 1.0
        std = 0.5
        spike_ts = 100

        self.actor = SpikingActor(obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                                  mean_range, std, spike_ts, device).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)  # Experience replay buffer



    def select_action(self, state, batch_size=1):
        state = torch.FloatTensor(state).to(self.device)
        mu, _ = self.actor(state, batch_size)
        # Sample from the distribution instead of taking the mean
        probs = F.softmax(mu, dim=-1)
        action = torch.multinomial(probs, 1).item() # Sample an action index
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_policy(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert experiences to tensors
        states = torch.FloatTensor(states).to(self.device)
        # Convert actions to one-hot encoding
        actions = F.one_hot(torch.LongTensor(actions), num_classes=self.act_dim).float().to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Calculate the actor loss (policy gradient)
        mu, _ = self.actor(states, batch_size)
        log_probs = torch.log(torch.sum(mu * actions, dim=-1, keepdim=True) + 1e-10)  # Avoid log(0)
        loss = - (log_probs * rewards).mean()  # Policy gradient loss

        # Optimize the actor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Main training loop
episode_rewards = []  # Initialize this list to store episode rewards

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Channel()
    agent = Agent(env.observation_space, env.n_actions, device)

    num_episodes = 500
    batch_size = 32
    update_frequency = 10 # How often to update the policy

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0  # Count steps within the episode

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            step_count += 1

            # Update the policy periodically
            if len(agent.replay_buffer) >= batch_size and step_count % update_frequency == 0:
                agent.update_policy(batch_size)

        episode_rewards.append(episode_reward)  # Append the episode reward
        print(f"Episode {episode}, Reward: {episode_reward}")



if __name__ == "__main__":
    train()




# Generate scatter plot after training
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(episode_rewards)), episode_rewards, color='blue', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Scatter Plot of Episode Rewards')
    plt.show()

# Generate histogram plot after training
    plt.figure(figsize=(10, 6))
    plt.hist(episode_rewards, bins=20, edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Rewards')
    plt.grid(True)
    plt.show()

    # Calculate average rewards over a sliding window
    window_size = 10
    avg_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')

    # Plot the average rewards
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (over {} episodes)'.format(window_size))
    plt.title('Training Progress - Average Reward')
    plt.grid(True)
    plt.show()








