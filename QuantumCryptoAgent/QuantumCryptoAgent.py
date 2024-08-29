# Necessary imports and initial setup
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import pandas as pd
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import neat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF
from collections import defaultdict
import pickle
from scipy import stats

from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# Import Dynex for quantum optimization
import dynex
from autoqubo import Binarization, SamplingCompiler, SearchSpace, Utils
import dimod
from meta.data_processor import DataProcessor
dynex.test()

# Ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Global environment variable
global_env = None

# Define the main classes and functions

class CellularAutomata(nn.Module):
    def __init__(self, size):
        super(CellularAutomata, self).__init__()
        self.size = size
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data.fill_(1/9)  # Simple averaging rule

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class HypergraphSNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_edges):
        super(HypergraphSNN, self).__init__()
        self.hgconv = HypergraphConv(in_channels, out_channels, use_attention=False)
        self.lif = snnLayer(out_channels)
        self.num_edges = num_edges

    def forward(self, x, hyperedge_index):
        if x.dim() == 3:
            x = x.squeeze(0)  # Remove batch dimension if present
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # Add feature dimension if missing

        num_nodes = x.size(0)
        hyperedge_index = hyperedge_index[:, hyperedge_index[0] < num_nodes]

        x = self.hgconv(x, hyperedge_index)
        x = self.lif(x)
        return x

    def get_weights(self):
        return self.hgconv.lin.weight.data

    def set_weights(self, new_weights):
        self.hgconv.lin.weight.data = new_weights

def initialize_recurrent_ppo(env):
    env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
    ppo_agent = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    return ppo_agent

class DRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, ca_size, latent_dim, hg_channels, num_edges, device, ppo_agent=None):
        super(DRLAgent, self).__init__()
        self.state_dim = state_dim
        self.ca = CellularAutomata(ca_size).to(device)
        self.vae = VAE(state_dim, latent_dim).to(device)
        self.hgsnn = HypergraphSNN(latent_dim, hg_channels, num_edges).to(device)
        self.fc = nn.Linear(hg_channels, action_dim).to(device)
        self.device = device
        self.num_edges = num_edges
        self.ppo_agent = ppo_agent  # Optional PPO agent

    def forward(self, x, hyperedge_index):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1, 1)
        x = self.ca(x)
        x = x.view(batch_size, -1)
        x, _, _ = self.vae(x)

        x = x.squeeze(0) if x.dim() == 3 else x

        x = self.hgsnn(x, hyperedge_index)
        return self.fc(x.view(batch_size, -1))

    def act(self, state):
      state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
      num_nodes = state_tensor.size(1)
      num_edges = min(num_nodes, self.num_edges)

      node_indices = torch.arange(num_nodes, device=self.device).repeat(num_edges)
      edge_indices = torch.arange(num_edges, device=self.device).repeat_interleave(num_nodes)
      hyperedge_index = torch.stack([node_indices, edge_indices], dim=0)

      with torch.no_grad():
          action_prob = self(state_tensor, hyperedge_index).squeeze().cpu().numpy()

      if self.ppo_agent:
          ppo_action, _ = self.ppo_agent.predict(state, deterministic=True)
          combined_action = (ppo_action + action_prob) / 2
      else:
          combined_action = action_prob
 
      return combined_action

class snnLayer(nn.Module):
    def __init__(self, input_size, beta=0.9, threshold=1.0):
        super(snnLayer, self).__init__()
        self.input_size = input_size
        self.beta = beta
        self.threshold = threshold
        self.surrogate_function = surrogate.fast_sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.membrane = nn.Parameter(torch.zeros(self.input_size))

    def forward(self, input_tensor):
        self.membrane.data = self.membrane.data.to(input_tensor.device)

        self.membrane.data = self.beta * self.membrane.data + input_tensor

        spike = self.surrogate_function(self.membrane - self.threshold)

        self.membrane.data = self.membrane.data * (1 - spike)

        return spike

def stdp(pre_spikes, post_spikes, weights, learning_rate=0.01):
    pre_spikes = pre_spikes.view(1, -1)
    post_spikes = post_spikes.view(1, -1)

    delta_t = post_spikes.unsqueeze(2) - pre_spikes.unsqueeze(1)
    dw = learning_rate * torch.exp(-torch.abs(delta_t) / 20) * torch.sign(delta_t)

    dw = dw.sum(dim=0)

    dw = dw.t()

    return weights + dw

class CryptoEnvWrapper(gym.Env):
    def __init__(self, config):
        super(CryptoEnvWrapper, self).__init__()
        self.env = CryptoEnv(config)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.env.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()

class CryptoEnv:
    def __init__(self, config, lookback=1, initial_capital=1e6,
                 buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99):
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = config['price_array']
        self.tech_array = config['tech_array']
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1
        self.asset_memory = [self.initial_total_asset]

        self.time = lookback-1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        self.env_name = 'MulticryptoEnv'
        self.state_dim = self.get_state_dim()
        self.action_dim = self.price_array.shape[1]
        self.if_discrete = False
        self.target_return = 10

    def reset(self) -> np.ndarray:
        self.time = self.lookback-1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.asset_memory = [self.initial_total_asset]

        state = self.get_state()
        self.state_dim = self.get_state_dim()
        return state

    def step(self, actions) -> (np.ndarray, float, bool, None):
        self.time += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        for index in np.where(actions < 0)[0]:
            if price[index] > 0:
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:
            if price[index] > 0:
                buy_num_shares = min(self.cash // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16
        self.total_asset = next_total_asset
        self.asset_memory.append(self.total_asset)
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        info = {
            'total_asset': self.total_asset,
            'cash': self.cash,
            'stocks': self.stocks,
            'current_price': self.current_price,
        }
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, info

    def get_state(self):
        state = np.hstack((self.cash * 2 ** -18, self.stocks * 2 ** -3))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time-i]
            normalized_tech_i = tech_i * 2 ** -15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
        return state

    def get_state_dim(self):
        return 1 + (self.price_array.shape[1] + self.tech_array.shape[1])*self.lookback

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(price, 10))
            action_norm_vector.append(1/((10)**x))

        action_norm_vector = np.asarray(action_norm_vector) * 10000
        self.action_norm_vector = np.asarray(action_norm_vector)

def create_neat_config(input_dim, output_dim):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat-config.ini')
    config.genome_config.num_inputs = input_dim
    config.genome_config.num_outputs = output_dim
    return config

def eval_genomes(genomes, config):
    global global_env
    ppo_agent = RecurrentPPO.load('ppo_agent')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = DRLAgent(state_dim=global_env.state_dim, action_dim=global_env.action_dim,
                         ca_size=28, latent_dim=global_env.state_dim, hg_channels=64,
                         num_edges=100, device=device, ppo_agent=ppo_agent)

        total_reward = 0
        state = global_env.reset()
        done = False

        while not done:
            # Pass historical data and ticker list to the act method
            historical_data = global_env.price_array  # Adjust as needed
            ticker_list = global_env.crypto_num       # Adjust as needed

            action_probs = agent.act(state, historical_data, ticker_list)
            neat_output = net.activate(state)

            combined_action = (action_probs + np.array(neat_output)) / 2

            next_state, reward, done, _ = global_env.step(combined_action)

            pre_spikes = agent.hgsnn.lif.membrane.detach()
            post_spikes = torch.tensor(action_probs, device=device).view(1, -1)
            weights = agent.hgsnn.get_weights()

            new_weights = stdp(pre_spikes, post_spikes, weights)
            agent.hgsnn.set_weights(new_weights)

            total_reward += reward
            state = next_state

        genome.fitness = total_reward

def save_models(final_agent, winner_genome, state_dim, action_dim):
    torch.save({
        'state_dict': final_agent.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'ppo_agent': final_agent.ppo_agent.save('ppo_agent') if final_agent.ppo_agent else None
    }, 'final_agent.pth')

    with open('winner_genome.pkl', 'wb') as f:
        pickle.dump((winner_genome, state_dim, action_dim), f)

def load_models(device, env):
    checkpoint = torch.load('final_agent.pth')
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    ppo_agent = RecurrentPPO.load('ppo_agent')
    if checkpoint['ppo_agent']:
        ppo_agent = RecurrentPPO.load('ppo_agent')
    final_agent = DRLAgent(state_dim=state_dim,
                           action_dim=action_dim,
                           ca_size=28,
                           latent_dim=min(20, state_dim),
                           hg_channels=64,
                           num_edges=100,
                           device=device, ppo_agent=ppo_agent)
    final_agent.load_state_dict(checkpoint['state_dict'])

    with open('winner_genome.pkl', 'rb') as f:
        winner_genome, _, _ = pickle.load(f)

    neat_config = create_neat_config(state_dim, action_dim)
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, neat_config)

    return final_agent, winner_net

def train_ppo_agent(ppo_agent, env, total_timesteps=10000):
    ppo_agent.learn(total_timesteps=total_timesteps)
    return ppo_agent

# Quantum action Optimization using Dynex
def quantum_action_optimization(state, possible_actions, historical_data, ticker_list=None, model="qpu"):
    """
    Use quantum optimization to select the best action for a given state.

    :param state: The current state of the environment
    :param possible_actions: List of possible actions
    :param historical_data: Historical data used for calculating the objective function
    :param ticker_list: List of tickers (not used in this function, but can be included if needed)
    :param model: The model type for Dynex (e.g., "qpu" for quantum processing unit)
    :return: The selected action
    """
    num_actions = len(possible_actions)

    def objective_function(action_index):
        action = possible_actions[action_index]

        # Calculate components of the objective
        expected_returns = calculate_expected_returns(action, state, historical_data)
        portfolio_variance = calculate_portfolio_variance(action, historical_data)
        liquidity_score = calculate_liquidity_score(action, state, historical_data)
        momentum_score = calculate_momentum_score(action, historical_data)
        diversification_score = calculate_diversification_score(action)
        transaction_costs = calculate_transaction_costs(action, state, transaction_cost=0.001)
        max_drawdown = calculate_max_drawdown(action, historical_data)

        # Calculate Sharpe ratio safely
        sharpe_ratio = safe_sharpe_ratio(expected_returns, portfolio_variance)

        # Combine scores into a single objective value
        score = (
            np.mean(expected_returns) * 0.3 -
            portfolio_variance * 0.2 +
            sharpe_ratio * 0.2 +
            liquidity_score * 0.1 +
            momentum_score * 0.1 +
            diversification_score * 0.1 -
            transaction_costs * 0.1 -
            max_drawdown * 0.2
        )

        return score

    # Create a QUBO matrix
    Q = {}
    for i in range(num_actions):
        Q[(i, i)] = -objective_function(i)  # Minimize the negative of the score to maximize it
        for j in range(i + 1, num_actions):
            Q[(i, j)] = 2  # Off-diagonal for one-hot encoding enforcement

    # Sanitize the QUBO matrix to remove NaN and Inf values
    Q = sanitize_qubo(Q)

    # Create a BQM from the QUBO
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    model = dynex.BQM(bqm)

    # Sample using the Dynex sampler
    sampler = dynex.DynexSampler(model, mainnet=False)
    sampleset = sampler.sample(num_reads=1000, annealing_time=10)

    # Get the best sample
    best_sample = sampleset.first.sample

    # Convert the binary solution to an action index
    selected_action_index = max(best_sample, key=best_sample.get)

    return possible_actions[selected_action_index]



def calculate_expected_returns(action, state, historical_data):
    # Calculate percentage changes manually using numpy
    returns = np.diff(historical_data, axis=0) / historical_data[:-1]

    # Calculate the mean of the last 30 days' returns
    returns = returns[-30:].mean(axis=0)

    # If action is a single value, returns should also be a single value
    if isinstance(action, (int, float)):
        return action * returns
    else:
        return np.dot(action, returns)


def calculate_portfolio_variance(action, historical_data):
    # Calculate percentage changes manually using numpy
    returns = np.diff(historical_data, axis=0) / historical_data[:-1]

    # Calculate the covariance matrix of returns
    cov_matrix = np.cov(returns.T)  # Use numpy's covariance function

    # Calculate the portfolio variance
    if isinstance(action, (int, float)):
        portfolio_variance = action ** 2 * cov_matrix
    else:
        portfolio_variance = np.dot(action, np.dot(cov_matrix, action))

    return portfolio_variance


def sanitize_qubo(Q):
    for key, value in Q.items():
        if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray)):
            # Replace NaN/Inf with 0 in arrays
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                Q[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Handle scalar values
            if np.isnan(value) or np.isinf(value):
                Q[key] = 0  # Replace NaN/Inf with 0 or another suitable fallback value
    return Q



def calculate_liquidity_score(action, state, historical_data):
    # Convert historical_data to a DataFrame
    historical_data_df = pd.DataFrame(historical_data)

    # Simple implementation: use average trading volume as a proxy for liquidity
    avg_volume = historical_data_df.rolling(window=30).mean().iloc[-1]

    if isinstance(action, (int, float)):
        liquidity_score = action * avg_volume / np.abs(action)
    else:
        liquidity_score = np.dot(action, avg_volume) / np.sum(np.abs(action))

    return liquidity_score




def calculate_momentum_score(action, historical_data):
    # Convert historical_data to a DataFrame
    historical_data_df = pd.DataFrame(historical_data)

    # Calculate 1-month and 3-month returns
    returns_1m = historical_data_df.pct_change(periods=20).iloc[-1]
    returns_3m = historical_data_df.pct_change(periods=60).iloc[-1]

    # Combine the two momentum signals
    combined_returns = (returns_1m + returns_3m) / 2

    if isinstance(action, (int, float)):
        momentum_score = action * combined_returns
    else:
        momentum_score = np.dot(action, combined_returns)

    return momentum_score





def calculate_diversification_score(action):
    if isinstance(action, (int, float)):
        diversification_score = 1  # A single asset has no diversification
    else:
        # Use the Herfindahl-Hirschman Index (HHI) as a measure of diversification
        action_weights = np.abs(action) / np.sum(np.abs(action))
        hhi = np.sum(action_weights ** 2)
        diversification_score = 1 - hhi  # Higher score means more diversified

    return diversification_score



def calculate_transaction_costs(action, state, transaction_cost):
    if isinstance(state, (int, float)):
        # If state is a single value, assume it's the current portfolio weight for a single asset
        current_weights = np.array([state])
    elif isinstance(action, (int, float)):
        # If action is a single value, assume we're working with a single asset
        current_weights = np.array([state[-1]])  # Assuming state has at least one element
    else:
        # If state is an array-like object, slice the last len(action) elements
        current_weights = state[-len(action):]

    # Calculate the transaction volume
    transaction_volume = np.sum(np.abs(action - current_weights))

    # Return the transaction costs
    return transaction_volume * transaction_cost

def safe_sharpe_ratio(expected_returns, portfolio_variance, risk_free_rate=0.02):
    # Avoid division by zero and handle NaN values
    if portfolio_variance <= 0 or np.isnan(portfolio_variance):
        return 0.0
    else:
        sharpe_ratio = (np.mean(expected_returns) - risk_free_rate) / np.sqrt(portfolio_variance)
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            return 0.0
        return sharpe_ratio



def calculate_max_drawdown(action, historical_data):
    # Calculate portfolio value over time
    portfolio_value = historical_data * action if isinstance(action, (int, float)) else np.dot(historical_data, action)

    # Calculate the maximum drawdown
    peak = np.maximum.accumulate(portfolio_value)

    # Prevent division by zero by setting peaks with zero value to NaN
    peak[peak == 0] = np.nan

    drawdown = (peak - portfolio_value) / peak

    # Handle NaN values by replacing them with zero
    drawdown = np.nan_to_num(drawdown, nan=0.0)

    max_drawdown = np.max(drawdown)

    return max_drawdown



# Quantum-assisted portfolio optimization using Dynex
def quantum_portfolio_optimization(data, symbols, trading_days=252, budget=100, A=1, B=-1, C=100, model="qpu"):
    # Calculate daily and annualized returns
    returns = data.pct_change().dropna()
    mean_daily_returns = returns.mean()
    mean_annual_returns = mean_daily_returns * trading_days

    # Calculate covariance matrix
    cov_matrix = returns.cov()
    n = len(symbols)
    cov_matrix_np = np.array(cov_matrix) * trading_days  # annualized
    mean_vector_np = np.array(mean_annual_returns)

    # Define the objective function and constraints
    def variance(x):
        return x @ cov_matrix_np @ x

    def mean(x):
        return x @ mean_vector_np

    def constraint(x):
        return (x.sum() - budget)**2

    def f(x):
        return A * variance(x) + B * mean(x) + C * constraint(x)

    # Define the search space
    s = SearchSpace()
    weights_vector = Binarization.get_uint_vector_type(n, n)
    s.add('x', weights_vector, n * n)

    # Generate the QUBO matrix
    qubo, offset = SamplingCompiler.generate_qubo_matrix(fitness_function=f, input_size=s.size, searchspace=s, use_multiprocessing=False)

    if not SamplingCompiler.test_qubo_matrix(f, qubo, offset, search_space=s):
        raise ValueError("QUBO generation failed - the objective function is not quadratic")

    # Solve the QUBO using Dynex
    sampleset = dynex.sample_qubo(qubo, offset, num_reads=10000, annealing_time=1000)

    # Extract the best solution
    sol = sampleset.record[0][0]
    x = s.decode_dict(sol)['x']

    # Convert solution to portfolio weights
    weights = np.array(x) / sum(x)

    return weights

def calculate_trading_days(start_date, end_date):
    # Calculate the total number of calendar days between start_date and end_date
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    return total_days


def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, env_class, if_vix=True, **kwargs):
    global global_env

    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list,
                                                        if_vix, cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    global_env = env_class(config=data_config)
    wrapped_env = CryptoEnvWrapper(config=data_config)
    state_dim = wrapped_env.observation_space.shape[0]
    action_dim = wrapped_env.action_space.shape[0]

    ppo_agent = initialize_recurrent_ppo(wrapped_env)
    ppo_agent = train_ppo_agent(ppo_agent, wrapped_env, total_timesteps=10000)
    ppo_agent.save('ppo_agent')

    neat_config = create_neat_config(state_dim, action_dim)

    pop = neat.Population(neat_config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 50)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_agent = DRLAgent(state_dim=state_dim,
                           action_dim=action_dim,
                           ca_size=28,
                           latent_dim=min(20, state_dim),
                           hg_channels=64,
                           num_edges=100,
                           device=device,
                           ppo_agent=ppo_agent)

    save_models(final_agent, winner, state_dim, action_dim)

    return final_agent, winner_net, ppo_agent, winner_net




def calculate_daily_returns(prices):
    prices = np.asarray(prices)
    if len(prices.shape) == 1:
        daily_returns = np.diff(prices) / prices[:-1]
    else:
        daily_returns = np.diff(prices, axis=0) / prices[:-1, :]
    return daily_returns

def calculate_cumulative_returns(daily_returns):
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    return cumulative_returns

def calculate_performance(values):
    return (values[-1] - values[0]) / values[0]

def calculate_buy_and_hold(price_array, initial_capital=1e6):
    daily_returns = calculate_daily_returns(price_array)
    cumulative_returns = calculate_cumulative_returns(daily_returns)

    total_return = cumulative_returns[-1]
    buy_hold_values = initial_capital * (1 + cumulative_returns)

    return total_return, buy_hold_values

def test(start_date, end_date, ticker_list, data_source, time_interval,
         technical_indicator_list, env_class, historical_start_date, if_vix=True, **kwargs):
    global global_env

    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list,
                                                        if_vix, cache=True)
    DP_historical = DataProcessor(data_source, historical_start_date, end_date, time_interval, **kwargs)
    historical_data, historical_tech_array, historical_turbulence_array  = DP_historical.run(ticker_list, technical_indicator_list, if_vix, cache=True)

    # Calculate the number of trading days (using calendar days)
    trading_days = calculate_trading_days(start_date, end_date)

    data = pd.DataFrame(historical_data, columns=ticker_list)

    data_config = {'price_array':price_array,
                   'tech_array':tech_array,
                   'turbulence_array':turbulence_array}

    global_env = env_class(config=data_config)

    neat_config = create_neat_config(global_env.state_dim, global_env.action_dim)
    final_agent, winner_net, ppo_agent, neat_agent = kwargs.get('trained_models', (None, None, None, None))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if final_agent is None or winner_net is None or ppo_agent is None or neat_agent is None:
        try:
            if os.path.exists("winner_genome.pkl") and os.path.exists("final_agent.pth"):
                final_agent, winner_net = load_models(device, global_env)
                ppo_agent = RecurrentPPO.load('ppo_agent')
                #neat_agent = neat.nn.FeedForwardNetwork.create(winner_net, neat_config)
            else:
                raise FileNotFoundError("winner_genome.pkl not found")
            print("Saved models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure that both 'final_agent.pth' and 'winner_genome.pkl' exist in the current directory.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred while loading the models: {e}")
            return None, None

    state = global_env.reset()
    done = False
    total_reward = 0
    ai_performance = [global_env.initial_total_asset]
    initial_total_asset = global_env.initial_total_asset

    neat_performance = [global_env.initial_total_asset]
    ppo_performance = [global_env.initial_total_asset]
    hybrid_performance = [global_env.initial_total_asset]
    final_agent_neat =  final_agent
    final_agent_neat.ppo_agent = None
    neat_done = False
    print("Testing NEAT model...")
    while not neat_done:

        action_probs = final_agent_neat.act(state)
        neat_output = winner_net.activate(state)
        ppo_action, _ = ppo_agent.predict(state, deterministic=True)
        combined_action = (action_probs + np.array(neat_output)) / 2
        # Step with NEAT
        neat_next_state, neat_reward, neat_done, _ = global_env.step(combined_action)
        neat_performance.append(global_env.total_asset)
        state = neat_next_state
    print("Testing with NEAT model completed")

    # Reset state and environment for PPO step
    global_env.reset()
    state = global_env.reset()
    ppo_done = False
    print("Testing with PPO agent ....")
    while not ppo_done:
        ppo_action, _ = ppo_agent.predict(state, deterministic=True)
        ppo_next_state, ppo_reward, ppo_done, _ = global_env.step(ppo_action)
        ppo_performance.append(global_env.total_asset)
        state = ppo_next_state
    print("Testing with PPO agent completed  ")

    # Reset state and environment for hybrid step
    global_env.reset()
    state = global_env.reset()
    print("Testing with Hybrid Model Started")
    # Define possible actions (discretize the action space)
    possible_actions = [-1, 0, 1]
    # Quantum-assisted portfolio optimization
    optimal_weights = quantum_portfolio_optimization(data, ticker_list, trading_days=trading_days, model="qpu")
    # Store the original act method
    original_act_method = final_agent.act
    # Apply optimal weights to the final agent's action
    def quantum_adjusted_action(state):
        original_action = original_act_method(state)  # Adjusted for new parameters
        # Use quantum optimization to select the best action
        quantum_action = quantum_action_optimization(state, possible_actions, historical_data, ticker_list)
        neat_output = winner_net.activate(state)
        # Combine the original, neat and quantum actions
        combined_actions = (original_action + quantum_action + np.array(neat_output)) / 3
        return combined_actions * optimal_weights
    # Replace the act method with the adjusted one
    final_agent.act = quantum_adjusted_action

    while not done:
        action_probs = final_agent.act(state)
        next_state, reward, done, _ = global_env.step(action_probs)
        hybrid_performance.append(global_env.total_asset)

        total_reward += reward
        state = next_state

        ai_performance.append(global_env.total_asset)
    print("Testing with Hybrid model completed")

    agent_return = calculate_performance(ai_performance)

    buy_and_hold_return, buy_hold_values = calculate_buy_and_hold(price_array, global_env.initial_cash)

    ai_returns = calculate_daily_returns(ai_performance)
    bh_returns = calculate_daily_returns(buy_hold_values)

    ai_sharpe = (np.mean(ai_returns) / np.std(ai_returns)) * np.sqrt(252)
    bh_sharpe = (np.mean(bh_returns) / np.std(bh_returns)) * np.sqrt(252)

    ai_max_drawdown = np.max(np.maximum.accumulate(ai_performance) - ai_performance) / np.max(np.maximum.accumulate(ai_performance))
    bh_max_drawdown = np.max(np.maximum.accumulate(buy_hold_values) - buy_hold_values) / np.max(np.maximum.accumulate(buy_hold_values))

    ai_volatility = np.std(ai_returns) * np.sqrt(252)
    bh_volatility = np.std(bh_returns) * np.sqrt(252)

    print(f"AI Trader Sharpe Ratio: {ai_sharpe:.2f}")
    print(f"Buy and Hold Sharpe Ratio: {bh_sharpe:.2f}")
    print(f"AI Trader Max Drawdown: {ai_max_drawdown:.2f}")
    print(f"Buy and Hold Max Drawdown: {bh_max_drawdown:.2f}")
    print(f"AI Trader Volatility: {ai_volatility:.2f}")
    print(f"Buy and Hold Volatility: {bh_volatility:.2f}")

    outperformance = agent_return - buy_and_hold_return
    print(f"Agent outperformance: {outperformance:.2%}")

    return {
        "total_reward": total_reward,
        "agent_return": agent_return,
        "buy_and_hold_return": buy_and_hold_return,
        "ai_performance": ai_performance,
        "buy_hold_performance": buy_hold_values,
        "neat_performance": neat_performance,
        "ppo_performance": ppo_performance,
        "hybrid_performance": hybrid_performance
    }

def plot_performance(results, start_date, end_date):
    plt.figure(figsize=(14, 8))

    ai_perf = np.array(results["ai_performance"])
    bh_perf = np.array(results["buy_hold_performance"])
    neat_perf = np.array(results["neat_performance"])
    ppo_perf = np.array(results["ppo_performance"])
    hybrid_perf = np.array(results["hybrid_performance"])

    ai_returns = (ai_perf - ai_perf[0]) / ai_perf[0] * 100
    bh_returns = (bh_perf - bh_perf[0]) / bh_perf[0] * 100
    neat_returns = (neat_perf - neat_perf[0]) / neat_perf[0] * 100
    ppo_returns = (ppo_perf - ppo_perf[0]) / ppo_perf[0] * 100
    hybrid_returns = (hybrid_perf - hybrid_perf[0]) / hybrid_perf[0] * 100

    date_range = pd.date_range(start=start_date, end=end_date, periods=len(ai_perf))

    plt.plot(date_range, ai_returns, label='AI Trader', color='blue', linestyle='-', linewidth=2)
    plt.plot(date_range, bh_returns, label='Buy and Hold', color='red', linestyle='--', linewidth=2)
    plt.plot(date_range, neat_returns, label='NEAT', color='green', linestyle='-.', linewidth=2)
    plt.plot(date_range, ppo_returns, label='PPO', color='purple', linestyle=':', linewidth=2)
    plt.plot(date_range, hybrid_returns, label='Hybrid', color='orange', linestyle='-', linewidth=2)

    plt.scatter(date_range[0], ai_returns[0], color='blue', marker='o', label='Start (AI Trader)')
    plt.scatter(date_range[0], bh_returns[0], color='red', marker='x', label='Start (Buy and Hold)')
    plt.scatter(date_range[0], neat_returns[0], color='green', marker='s', label='Start (NEAT)')
    plt.scatter(date_range[0], ppo_returns[0], color='purple', marker='d', label='Start (PPO)')
    plt.scatter(date_range[0], hybrid_returns[0], color='orange', marker='^', label='Start (Hybrid)')

    ai_peak_idx = np.argmax(ai_returns)
    bh_peak_idx = np.argmax(bh_returns)
    neat_peak_idx = np.argmax(neat_returns)
    ppo_peak_idx = np.argmax(ppo_returns)
    hybrid_peak_idx = np.argmax(hybrid_returns)

    plt.scatter(date_range[ai_peak_idx], ai_returns[ai_peak_idx], color='blue', marker='^', label='Peak (AI Trader)', s=100)
    plt.scatter(date_range[bh_peak_idx], bh_returns[bh_peak_idx], color='red', marker='v', label='Peak (Buy and Hold)', s=100)
    plt.scatter(date_range[neat_peak_idx], neat_returns[neat_peak_idx], color='green', marker='*', label='Peak (NEAT)', s=100)
    plt.scatter(date_range[ppo_peak_idx], ppo_returns[ppo_peak_idx], color='purple', marker='*', label='Peak (PPO)', s=100)
    plt.scatter(date_range[hybrid_peak_idx], hybrid_returns[hybrid_peak_idx], color='orange', marker='*', label='Peak (Hybrid)', s=100)

    plt.annotate(f'Start: {ai_returns[0]:.2f}%', (date_range[0], ai_returns[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    plt.annotate(f'Start: {bh_returns[0]:.2f}%', (date_range[0], bh_returns[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')
    plt.annotate(f'Start: {neat_returns[0]:.2f}%', (date_range[0], neat_returns[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='green')
    plt.annotate(f'Start: {ppo_returns[0]:.2f}%', (date_range[0], ppo_returns[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='purple')
    plt.annotate(f'Start: {hybrid_returns[0]:.2f}%', (date_range[0], hybrid_returns[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='orange')

    plt.annotate(f'End: {ai_returns[-1]:.2f}%', (date_range[-1], ai_returns[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    plt.annotate(f'End: {bh_returns[-1]:.2f}%', (date_range[-1], bh_returns[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')
    plt.annotate(f'End: {neat_returns[-1]:.2f}%', (date_range[-1], neat_returns[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='green')
    plt.annotate(f'End: {ppo_returns[-1]:.2f}%', (date_range[-1], ppo_returns[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='purple')
    plt.annotate(f'End: {hybrid_returns[-1]:.2f}%', (date_range[-1], hybrid_returns[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='orange')

    plt.fill_between(date_range, ai_returns, bh_returns, where=(ai_returns > bh_returns), color='blue', alpha=0.1, label='AI Outperformance')
    plt.fill_between(date_range, ai_returns, bh_returns, where=(ai_returns < bh_returns), color='red', alpha=0.1, label='BH Outperformance')

    plt.title('Performance Comparison: AI Trader vs NEAT vs PPO vs Hybrid vs Buy and Hold', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Return (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    TRAIN_START_DATE = "2019-01-01"
    TRAIN_END_DATE = "2023-12-31"
    TEST_START_DATE = "2023-11-22"
    TEST_END_DATE = "2024-07-31"
    TICKER_LIST = ["SOLUSDT"]
    TIME_INTERVAL = "1d"
    TECHNICAL_INDICATORS = ["macd", "rsi", "cci", "dx"]

    # trained_agent, trained_neat, ppo_agent, neat_agent = train(TRAIN_START_DATE, TRAIN_END_DATE, TICKER_LIST, 'binance', TIME_INTERVAL,
    #                                                            TECHNICAL_INDICATORS, CryptoEnv, if_vix=False)

    #print("Models saved successfully.")
    trained_agent = None
    trained_neat = None
    ppo_agent = None
    neat_agent = None


    results = test(TEST_START_DATE, TEST_END_DATE, TICKER_LIST, 'binance', TIME_INTERVAL,
                   TECHNICAL_INDICATORS, CryptoEnv,TRAIN_START_DATE, if_vix=False,
                   trained_models=(trained_agent, trained_neat, ppo_agent, neat_agent))

    print(f"Test episode finished. Total reward: {results['total_reward']}")
    print(f"Agent return: {results['agent_return']}")
    print(f"Buy-and-hold return: {results['buy_and_hold_return']}")

    outperformance = results['agent_return'] - results['buy_and_hold_return']
    print(f"Agent outperformance: {outperformance:.2%}")

    plot_performance(results, TEST_START_DATE, TEST_END_DATE)

