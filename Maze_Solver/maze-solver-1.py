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

# VAE Constants
LATENT_DIM = 10

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

    def update(self, input_current):
        if self.refractory_time > 0:
            self.refractory_time -= 1
            return 0

        self.membrane_potential += input_current
        if self.membrane_potential >= THRESHOLD:
            self.has_spiked = True
            self.membrane_potential = 0
            self.refractory_time = REFRACTORY_PERIOD
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
        spikes = [neuron.update(output) for neuron, output in zip(self.neurons, neat_output)]
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

def eval_genomes(genomes, config):
    print(f"Evaluating {len(genomes)} genomes")
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    print("New maze generated")
    input_dim = 6  # Increased input dimension

    font = pygame.font.SysFont(None, 30)

    maze_surface = pygame.Surface((MAZE_SURFACE_WIDTH, MAZE_SURFACE_HEIGHT))
    info_surface = pygame.Surface((WIN_WIDTH, WIN_HEIGHT - MAZE_SURFACE_HEIGHT))

    for i, (genome_id, genome) in enumerate(genomes):
        print(f"Evaluating genome {i+1}/{len(genomes)}")
        net = NEATCAHypergraph(genome, config, input_dim)
        agent_pos = list(maze.start)
        steps = 0
        max_steps = MAZE_WIDTH * MAZE_HEIGHT * 2
        last_positions = []
        visited = set()

        while steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            maze_surface.fill(WHITE)
            info_surface.fill(WHITE)

            maze.draw(maze_surface)
            pygame.draw.rect(maze_surface, BLUE, (agent_pos[0] * CELL_SIZE, agent_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Get inputs
            inputs = [
                agent_pos[0] / MAZE_WIDTH,
                agent_pos[1] / MAZE_HEIGHT,
                (maze.end[0] - agent_pos[0]) / MAZE_WIDTH,
                (maze.end[1] - agent_pos[1]) / MAZE_HEIGHT,
                len(visited) / (MAZE_WIDTH * MAZE_HEIGHT),  # Exploration factor
                steps / max_steps  # Time factor
            ]

            output = net.activate(inputs)

            if len(output) < 4:
                output = np.pad(output, (0, 4 - len(output)))
            elif len(output) > 4:
                output = output[:4]

            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            valid_moves = []

            for idx, (dx, dy) in enumerate(moves):
                new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
                if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and
                    maze.maze[new_y][new_x] == 0):
                    # Encourage exploration of unvisited cells
                    exploration_bonus = 0.1 if (new_x, new_y) not in visited else 0
                    valid_moves.append((idx, output[idx] + exploration_bonus))

            if valid_moves:
                move_direction, _ = max(valid_moves, key=lambda x: x[1])
                dx, dy = moves[move_direction]
                new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
                
                # Prevent immediate backtracking
                if len(last_positions) > 1 and [new_x, new_y] == last_positions[-2]:
                    valid_moves.remove((move_direction, _))
                    if valid_moves:
                        move_direction, _ = max(valid_moves, key=lambda x: x[1])
                        dx, dy = moves[move_direction]
                        new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
                    else:
                        # If no other valid moves, stay in place
                        new_x, new_y = agent_pos

                agent_pos = [new_x, new_y]
                steps += 1
                visited.add((new_x, new_y))
                last_positions.append(agent_pos)
                if len(last_positions) > 5:
                    last_positions.pop(0)
                print(f"Valid move: Agent moved to {agent_pos}")
            else:
                steps += 1
                print(f"No valid move available")

            if tuple(agent_pos) == maze.end:
                print("Agent reached the end!")
                break

            gen_text = font.render(f"Generation: {p.generation+1}, Genome: {i+1}/{len(genomes)}", True, BLACK)
            info_surface.blit(gen_text, (10, 10))

            WIN.blit(maze_surface, (0, 0))
            WIN.blit(info_surface, (0, MAZE_SURFACE_HEIGHT))

            pygame.display.flip()
            time.sleep(0.1)

        distance_to_end = ((agent_pos[0] - maze.end[0])**2 + (agent_pos[1] - maze.end[1])**2)**0.5
        fitness = 1 / (distance_to_end + 1)

        if tuple(agent_pos) == maze.end:
            fitness += 10

        # Reward exploration
        fitness += len(visited) / (MAZE_WIDTH * MAZE_HEIGHT)

        # Penalize steps, but less severely
        fitness -= steps / (MAZE_WIDTH * MAZE_HEIGHT * 8)

        genome.fitness = max(0, fitness)
        print(f"Genome {i+1} fitness: {genome.fitness}")

def run_maze_solver():
    pygame.init()
    pygame.font.init()

    global WIN
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Maze Solver SNN")

    config_path = os.path.join(os.path.dirname(__file__), 'neat_config_snn.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    global p
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(winner))

    print('\nFinal Statistics')
    stats.save()

    pygame.quit()

if __name__ == "__main__":
    run_maze_solver()