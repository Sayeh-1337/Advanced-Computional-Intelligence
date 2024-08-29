import pygame
import neat
import random
import os
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import time
import pickle
import math

# Initialize Pygame
pygame.init()

# Initialize the font module
pygame.font.init()

# Game Constants
WIN_WIDTH = 1000
WIN_HEIGHT = 700
CELL_SIZE = 20
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
MAZE_SURFACE_WIDTH = MAZE_WIDTH * CELL_SIZE
MAZE_SURFACE_HEIGHT = MAZE_HEIGHT * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up the display
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Maze Solver SNN")

# SNN Constants
NUM_NEURONS = 100
THRESHOLD = 1.0
REFRACTORY_PERIOD = 5
TIME_STEP = 0.1

#AGENT CONSTANTS
RANDOM_MOVE_CHANCE = 0.1
WALL_FOLLOWING_CHANCE = 0.2
MEMORY_SIZE = 10
GOAL_PROXIMITY_THRESHOLD = 2

# VAE Constants
LATENT_DIM = 10

# Exploration Constants
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.1
ANNEALING_STEPS = 1000

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
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

class SpikingNeuron:
    def __init__(self):
        self.membrane_potential = 0
        self.refractory_time = 0
        self.has_spiked = False
        self.last_spike_time = -1000  # Initialize to a large negative value

    def update(self, input_current, current_time):
        if self.refractory_time > 0:
            self.refractory_time -= 1
            return 0

        self.membrane_potential += input_current
        if self.membrane_potential >= THRESHOLD:
            self.has_spiked = True
            self.membrane_potential = 0
            self.refractory_time = REFRACTORY_PERIOD
            self.last_spike_time = current_time
            return 1
        return 0

class CellularAutomaton:
    def __init__(self, size):
        self.size = size
        self.state = np.random.randint(0, 2, size)

    def update(self, inputs):
        inputs_resized = np.resize(inputs, self.size)
        self.state = (self.state + inputs_resized) % 2

    def get_state(self):
        return self.state

class NEATCAHypergraph:
    def __init__(self, genome, config, input_dim):
        self.genome = genome
        self.config = config
        self.neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.vae = VAE(input_dim, LATENT_DIM)
        self.ca = CellularAutomaton(NUM_NEURONS)
        self.neurons = [SpikingNeuron() for _ in range(NUM_NEURONS)]
        self.hypergraph = hnx.Hypergraph({})
        self.weights = {}
        self.last_spike_times = {i: -1000 for i in range(NUM_NEURONS)}
        self.current_time = 0
    
    def stdp_update(self, pre_neuron, post_neuron):
        t_diff = self.last_spike_times[post_neuron] - self.last_spike_times[pre_neuron]
        
        # STDP parameters
        A_plus = 0.1
        A_minus = -0.12
        tau_plus = 20
        tau_minus = 20

        if t_diff > 0:
            dw = A_plus * math.exp(-t_diff / tau_plus)
        else:
            dw = A_minus * math.exp(t_diff / tau_minus)
        
        # Update weight
        connection_key = (pre_neuron, post_neuron)
        if connection_key in self.weights:
            self.weights[connection_key] += dw
            # Clip weight to keep it within a reasonable range
            self.weights[connection_key] = max(-1, min(1, self.weights[connection_key]))

    def activate(self, inputs):
        with torch.no_grad():
            input_tensor = torch.FloatTensor(inputs)
            mu, _ = self.vae.encode(input_tensor)
            encoded = mu.numpy()

        self.ca.update(encoded)
        ca_state = self.ca.get_state()
        combined_input = np.concatenate([encoded, ca_state])

        expected_input_size = len(self.neat_network.input_nodes)
        if len(combined_input) > expected_input_size:
            combined_input = combined_input[:expected_input_size]
        elif len(combined_input) < expected_input_size:
            combined_input = np.pad(combined_input, (0, expected_input_size - len(combined_input)))

        neat_output = self.neat_network.activate(combined_input)
        spikes = [neuron.update(output, self.current_time) for neuron, output in zip(self.neurons, neat_output)]
        # Apply STDP
        for i, spike in enumerate(spikes):
            if spike:
                for j in range(NUM_NEURONS):
                    if i != j:
                        self.stdp_update(j, i)
        
        self.current_time += 1
        self.update_hypergraph(inputs, encoded, spikes)

        return spikes

    def update_hypergraph(self, inputs, encoded_state, spikes):
        edges = {}

        for i, inp in enumerate(inputs):
            edges[f'input_{i}'] = {f'input_{i}'}

        for i, neuron in enumerate(self.neurons):
            edges[f'neuron_{i}'] = {f'neuron_{i}'}

        for i, spike in enumerate(spikes):
            edges[f'output_{i}'] = {f'output_{i}'}

        for cg in self.genome.connections.values():
            if cg.enabled:
                input_node = f'neuron_{cg.key[0]}'
                output_node = f'neuron_{cg.key[1]}'
                edge_name = f'connection_{cg.key[0]}_{cg.key[1]}'
                edges[edge_name] = {input_node, output_node}

        edges['encoded_state'] = set([f'neuron_{i}' for i in range(NUM_NEURONS)])
        self.hypergraph = hnx.Hypergraph(edges)

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = self.generate_maze()
        self.start = (1, 1)
        self.end = (width - 2, height - 2)

    def generate_maze(self):
        maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        stack = [(1, 1)]
        while stack:
            x, y = stack[-1]
            maze[y][x] = 0
            neighbors = [(x+2, y), (x-2, y), (x, y+2), (x, y-2)]
            unvisited = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.width and 0 <= ny < self.height and maze[ny][nx] == 1]
            if unvisited:
                nx, ny = random.choice(unvisited)
                maze[ny][nx] = 0
                maze[(y + ny) // 2][(x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def draw(self, surface):
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, BLACK if cell else WHITE, rect)

        start_rect = pygame.Rect(self.start[0] * CELL_SIZE, self.start[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        end_rect = pygame.Rect(self.end[0] * CELL_SIZE, self.end[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GREEN, start_rect)
        pygame.draw.rect(surface, RED, end_rect)

    def get_surrounding_walls(self, x, y):
        walls = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                walls.append(int(self.maze[ny][nx] == 1))
            else:
                walls.append(1)  # Consider out-of-bounds as walls
        return walls

import numpy as np

class Agent:
    def __init__(self, start_pos, net):
        self.pos = list(start_pos)
        self.memory = []
        self.net = net
        self.last_move = None
        self.move_cooldown = 0
        self.momentum = [0, 0]
        self.closest_to_goal = float('inf')  # Track closest distance to goal

    def update_memory(self):
        self.memory.append(tuple(self.pos))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def move(self, direction):
        self.pos[0] += direction[0]
        self.pos[1] += direction[1]
        self.update_memory()
        self.last_move = direction
        self.move_cooldown = 2
        # Reduce momentum influence
        self.momentum = [0.5 * self.momentum[0] + 0.5 * direction[0],
                         0.5 * self.momentum[1] + 0.5 * direction[1]]

def get_goal_proximity_threshold(distance_to_goal, maze_diagonal):
    return max(2, min(distance_to_goal * 0.5, maze_diagonal * 0.1))

def eval_genomes(genomes, config):
    print(f"Evaluating {len(genomes)} genomes")
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    print("New maze generated")
    input_dim = 15  # Increased input dimension for additional features

    font = pygame.font.SysFont(None, 30)

    maze_surface = pygame.Surface((MAZE_SURFACE_WIDTH, MAZE_SURFACE_HEIGHT))
    info_surface = pygame.Surface((WIN_WIDTH, WIN_HEIGHT - MAZE_SURFACE_HEIGHT))

    maze_diagonal = ((MAZE_WIDTH ** 2) + (MAZE_HEIGHT ** 2)) ** 0.5

    for i, (genome_id, genome) in enumerate(genomes):
        print(f"Evaluating genome {i+1}/{len(genomes)}")
        net = NEATCAHypergraph(genome, config, input_dim)
        agent = Agent(maze.start, net)
        steps = 0
        max_steps = MAZE_WIDTH * MAZE_HEIGHT * 2
        visited = set()
        start_revisits = 0

        start_time = time.time()  # Start the timer

        while steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            maze_surface.fill(WHITE)
            info_surface.fill(WHITE)

            maze.draw(maze_surface)
            pygame.draw.rect(maze_surface, BLUE, (agent.pos[0] * CELL_SIZE, agent.pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Get inputs
            surrounding_walls = maze.get_surrounding_walls(agent.pos[0], agent.pos[1])
            distance_to_end = ((agent.pos[0] - maze.end[0])**2 + (agent.pos[1] - maze.end[1])**2)**0.5
            agent.closest_to_goal = min(agent.closest_to_goal, distance_to_end)
            
            # Dynamic thresholding
            goal_proximity_threshold = get_goal_proximity_threshold(distance_to_end, maze_diagonal)
            goal_proximity_threshold = min(5, goal_proximity_threshold * 1.5)

            inputs = [
                agent.pos[0] / MAZE_WIDTH,
                agent.pos[1] / MAZE_HEIGHT,
                (maze.end[0] - agent.pos[0]) / MAZE_WIDTH,
                (maze.end[1] - agent.pos[1]) / MAZE_HEIGHT,
                len(visited) / (MAZE_WIDTH * MAZE_HEIGHT),
                steps / max_steps,
                agent.momentum[0],
                agent.momentum[1],
                distance_to_end / maze_diagonal,
                start_revisits / steps if steps > 0 else 0,
                1 / (1 + distance_to_end),  # Inverse distance to emphasize proximity
            ] + surrounding_walls

            output = net.activate(inputs)

            # Apply STDP
            for xx in range(len(net.neurons)):
                for yy in range(len(net.neurons)):
                    if xx != yy:
                        net.stdp_update(xx, yy)

            if len(output) < 4:
                output = np.pad(output, (0, 4 - len(output)))
            elif len(output) > 4:
                output = output[:4]

            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            valid_moves = []

            # Check if the goal is in an adjacent cell
            goal_adjacent = False
            for idx, (dx, dy) in enumerate(moves):
                new_x, new_y = agent.pos[0] + dx, agent.pos[1] + dy
                if (new_x, new_y) == maze.end:
                    goal_adjacent = True
                    goal_direction = idx
                    break

            if goal_adjacent:
                # If the goal is adjacent, move directly to it
                move_direction = goal_direction
            else:
                # Existing logic for choosing a move
                if random.random() < RANDOM_MOVE_CHANCE:
                    move_direction = random.randint(0, 3)
                elif random.random() < WALL_FOLLOWING_CHANCE:
                    wall_directions = [i for i, wall in enumerate(surrounding_walls) if wall == 1]
                    if wall_directions:
                        move_direction = random.choice(wall_directions)
                    else:
                        move_direction = np.argmax(output)
                else:
                    for idx, (dx, dy) in enumerate(moves):
                        new_x, new_y = agent.pos[0] + dx, agent.pos[1] + dy
                        if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and
                            (maze.maze[new_y][new_x] == 0 or (new_x, new_y) == maze.end)):  # Allow moving to the goal
                            exploration_bonus = 0.2 if (new_x, new_y) not in visited else 0
                            memory_penalty = 0.3 if (new_x, new_y) in agent.memory else 0
                            distance_to_end_new = ((new_x - maze.end[0])**2 + (new_y - maze.end[1])**2)**0.5
                            pathfinding_bonus = (distance_to_end - distance_to_end_new) / maze_diagonal
                            momentum_bonus = 0.1 * (agent.momentum[0] * dx + agent.momentum[1] * dy)
                            curiosity_bonus = 0.2 if (new_x, new_y) not in visited else 0
                            
                            move_score = (output[idx] + exploration_bonus - memory_penalty + 
                                          pathfinding_bonus + momentum_bonus + curiosity_bonus)
                            valid_moves.append((idx, move_score))

                    if valid_moves:
                        move_direction, _ = max(valid_moves, key=lambda x: x[1])
                    else:
                        move_direction = np.argmax(output)

            # Make the move
            new_x = agent.pos[0] + moves[move_direction][0]
            new_y = agent.pos[1] + moves[move_direction][1]

            # Check if the new position is valid (including the goal)
            if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and
                (maze.maze[new_y][new_x] == 0 or (new_x, new_y) == maze.end) and
                agent.move_cooldown == 0):
                agent.move(moves[move_direction])
                steps += 1
                visited.add(tuple(agent.pos))
                if tuple(agent.pos) == maze.start:
                    start_revisits += 1
                print(f"Valid move: Agent moved to {agent.pos}")
            else:
                steps += 1
                print(f"Invalid move: Agent stayed at {agent.pos}")

            if agent.move_cooldown > 0:
                agent.move_cooldown -= 1

            # Check if the agent has reached the goal
            if tuple(agent.pos) == maze.end:
                end_time = time.time()  # Stop the timer
                solve_time = end_time - start_time
                print(f"Goal reached in {steps} steps and {solve_time:.2f} seconds!")
                break

            gen_text = font.render(f"Generation: {p.generation+1}, Genome: {i+1}/{len(genomes)}", True, BLACK)
            time_text = font.render(f"Time: {time.time() - start_time:.2f} s", True, BLACK)
            info_surface.blit(gen_text, (10, 10))
            info_surface.blit(time_text, (10, 40))

            WIN.blit(maze_surface, (0, 0))
            WIN.blit(info_surface, (0, MAZE_SURFACE_HEIGHT))

            pygame.display.flip()
            time.sleep(0.1)

        if steps >= max_steps:
            end_time = time.time()  # Stop the timer if max steps reached
            solve_time = end_time - start_time
            print(f"Max steps reached. Time taken: {solve_time:.2f} seconds")

        # Fitness calculation (unchanged)
        distance_to_end = ((agent.pos[0] - maze.end[0])**2 + (agent.pos[1] - maze.end[1])**2)**0.5
        progress = 1 - (distance_to_end / maze_diagonal)
        
        fitness = progress * 30

        if tuple(agent.pos) == maze.end:
            fitness += 200  # Significantly increase reward for reaching the end

        # Reward exploration
        exploration_ratio = len(visited) / (MAZE_WIDTH * MAZE_HEIGHT)
        fitness += exploration_ratio * 20

        # Penalize steps, but less severely
        fitness -= steps / (MAZE_WIDTH * MAZE_HEIGHT * 20)

        # Penalize revisiting the start position
        fitness -= start_revisits * 5

        # Reward getting close to the goal
        fitness += (1 / (1 + agent.closest_to_goal)) * 50

        genome.fitness = max(0, fitness)
        print(f"Genome {i+1} fitness: {genome.fitness}")

def run_maze_solver(mode='train'):
    pygame.init()
    pygame.font.init()

    global WIN
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Maze Solver SNN")

    config_path = os.path.join(os.path.dirname(__file__), 'neat_config_snn.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if mode == 'train':
        global p
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(eval_genomes, 100)

        print('\nBest genome:\n{!s}'.format(winner))

        # Save the winner
        with open('maze_winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

        # Visualize the winner's neural network
        #visualize_winner(winner, config)

    elif mode == 'play':
        # Load the winner
        with open('maze_winner.pkl', 'rb') as input_file:
            winner = pickle.load(input_file)

        # Create the maze
        maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)

        # Create the network from the winner genome
        net = NEATCAHypergraph(winner, config, 15)  # 15 is the input dimension

        # Run the simulation with the winner
        agent_pos = list(maze.start)
        steps = 0
        max_steps = MAZE_WIDTH * MAZE_HEIGHT * 2
        font = pygame.font.SysFont(None, 30)

        start_time = time.time()  # Start the timer

        while steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get inputs
            surrounding_walls = maze.get_surrounding_walls(agent_pos[0], agent_pos[1])
            distance_to_end = ((agent_pos[0] - maze.end[0])**2 + (agent_pos[1] - maze.end[1])**2)**0.5
            maze_diagonal = ((MAZE_WIDTH ** 2) + (MAZE_HEIGHT ** 2)) ** 0.5

            inputs = [
                agent_pos[0] / MAZE_WIDTH,
                agent_pos[1] / MAZE_HEIGHT,
                (maze.end[0] - agent_pos[0]) / MAZE_WIDTH,
                (maze.end[1] - agent_pos[1]) / MAZE_HEIGHT,
                0,  # Exploration factor (not used in play mode)
                steps / max_steps,  # Time factor
                0, 0,  # Momentum (not used in play mode)
                distance_to_end / maze_diagonal,
                0,  # Start revisits (not used in play mode)
                1 / (1 + distance_to_end),  # Inverse distance to emphasize proximity
            ] + surrounding_walls

            output = net.activate(inputs)

            move_direction = np.argmax(output)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = moves[move_direction]
            new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy

            if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and
                maze.maze[new_y][new_x] == 0):
                agent_pos = [new_x, new_y]
                steps += 1

            # Draw everything
            WIN.fill(WHITE)
            maze.draw(WIN)
            pygame.draw.rect(WIN, BLUE, (agent_pos[0] * CELL_SIZE, agent_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            current_time = time.time() - start_time
            time_text = font.render(f"Time: {current_time:.2f} s", True, BLACK)
            steps_text = font.render(f"Steps: {steps}", True, BLACK)
            WIN.blit(time_text, (10, MAZE_SURFACE_HEIGHT + 10))
            WIN.blit(steps_text, (10, MAZE_SURFACE_HEIGHT + 40))

            pygame.display.flip()

            if tuple(agent_pos) == maze.end:
                end_time = time.time()
                solve_time = end_time - start_time
                print(f"Agent reached the end in {steps} steps and {solve_time:.2f} seconds!")
                break

            time.sleep(0.1)

        if steps >= max_steps:
            end_time = time.time()
            solve_time = end_time - start_time
            print(f"Agent couldn't reach the end within the step limit. Time taken: {solve_time:.2f} seconds")

        # Keep the window open for a few seconds after completion
        pygame.time.wait(3000)

if __name__ == "__main__":
    mode = input("Enter 'train' to train a new solver, or 'play' to watch the best solver: ").lower()
    while mode not in ['train', 'play']:
        mode = input("Invalid input. Please enter 'train' or 'play': ").lower()
    run_maze_solver(mode)