import pygame
import neat
import random
import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import pickle
import configparser
import math 

# Load configuration
config = configparser.ConfigParser()
config.read('simulation_config.ini')

# Game Constants
WIN_WIDTH = config.getint('Display', 'WIN_WIDTH')
WIN_HEIGHT = config.getint('Display', 'WIN_HEIGHT')
FLOOR = config.getint('Display', 'FLOOR')
DRAW_LINES = config.getboolean('Display', 'DRAW_LINES')

# CA and VAE Constants
CA_SIZE = config.getint('AI', 'CA_SIZE')
LATENT_DIM = config.getint('AI', 'LATENT_DIM')

# Hypergraph visualization constants
HYPERGRAPH_WIDTH = config.getint('Hypergraph', 'WIDTH')
HYPERGRAPH_HEIGHT = config.getint('Hypergraph', 'HEIGHT')
SHOW_HYPERGRAPH = config.getboolean('Hypergraph', 'SHOW')

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("3D UAV Control")
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

# Define GEN as a global variable
GEN = 0

# Increased velocity factor
VELOCITY_FACTOR = 3.0

# Increase number of obstacles
NUM_OBSTACLES = 10

# Room
ROOM_WIDTH = WIN_WIDTH - 200
ROOM_LENGTH = WIN_HEIGHT
ROOM_HEIGHT = 200

# New constants for home and target positions
HOME_POS = (50, 50, 50)
TARGET_POS = (ROOM_WIDTH - 50, ROOM_LENGTH - 50, 50)

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

class SNNLayerSTDP(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(SNNLayerSTDP, self).__init__()
        self.lif = snn.Leaky(beta=0.95)
        self.fc = nn.Linear(input_size, output_size)
        self.spike_grad = surrogate.fast_sigmoid()
        self.stdp_window = 20
        self.learning_rate = 0.01
        self.a_plus = 0.1
        self.a_minus = -0.105
        self.tau = 20.0
        self.spike_history = []
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        mem = self.lif.init_leaky()
        x = x.view(x.size(0), -1)  # Flatten the input
        spk, mem = self.lif(self.fc(x), mem)
        self.spike_history.append((x, spk))
        if len(self.spike_history) > self.stdp_window:
            self.spike_history.pop(0)
        return spk, mem

    def stdp_update(self):
        if len(self.spike_history) < 2:
            return

        pre_spikes, post_spikes = zip(*self.spike_history[-2:])
        pre_spikes = torch.cat(pre_spikes, dim=0)
        post_spikes = torch.cat(post_spikes, dim=0)

        dt = torch.arange(len(self.spike_history) - 1, -1, -1).float()
        
        weight_update = torch.zeros_like(self.fc.weight)
        for t in dt:
            pre_mask = (pre_spikes > 0).float()
            post_mask = (post_spikes > 0).float()
            
            weight_update += self.a_plus * torch.outer(post_mask.mean(0), pre_mask.mean(0)) * torch.exp(-t / self.tau)
            weight_update -= self.a_minus * torch.outer(pre_mask.mean(0), post_mask.mean(0)) * torch.exp(-t / self.tau)

        with torch.no_grad():
            self.fc.weight += self.learning_rate * weight_update

    def reset_spike_history(self):
        self.spike_history = []

class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(HypergraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        num_nodes = x.size(0)
        num_edges = hyperedge_index.size(1)

        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_edges, device=x.device)

        if hyperedge_index.shape[0] != 2:
            raise ValueError(f"hyperedge_index should have shape [2, num_edges], but got {hyperedge_index.shape}")

        node_degrees = torch.bincount(hyperedge_index[0], minlength=num_nodes).float()
        edge_degrees = torch.bincount(hyperedge_index[1], minlength=num_edges).float()

        node_deg_inv_sqrt = node_degrees.pow(-0.5)
        edge_deg_inv = 1.0 / (edge_degrees + 1e-5)
        norm = node_deg_inv_sqrt[hyperedge_index[0]] * edge_deg_inv[hyperedge_index[1]]

        sparse_tensor = torch.sparse_coo_tensor(hyperedge_index, norm * hyperedge_weight, (num_nodes, num_nodes))

        out = torch.mm(x, self.weight)

        if num_nodes == 1:
            return out

        sparse_out = torch.sparse.mm(sparse_tensor, out)

        return sparse_out

class NEATCAHypergraphSNN:
    def __init__(self, genome, config, vae):
        self.genome = genome
        self.config = config
        self.vae = vae
        
        self.input_size = 22  # 6 for UAV state + 4 * 3 for obstacles
        self.output_size = 3  # 3 for UAV control (vel_x, vel_y, vel_z)
        
        self.snn_layer = SNNLayerSTDP(self.input_size, self.output_size)
        self.hypergraph = nx.Graph()

    def activate(self, inputs):
        x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        
        spk, _ = self.snn_layer(x)
        
        self.update_hypergraph(x, spk)
        
        return spk.squeeze().tolist()

    def update_hypergraph(self, inputs, output):
        if self.hypergraph.number_of_nodes() > 100:
            nodes_to_remove = list(self.hypergraph.nodes())[:10]
            self.hypergraph.remove_nodes_from(nodes_to_remove)

        input_node = max(self.hypergraph.nodes()) + 1 if self.hypergraph.nodes() else 0
        output_node = input_node + 1

        self.hypergraph.add_nodes_from([(input_node, {"type": "input"}),
                                        (output_node, {"type": "output"})])

        self.hypergraph.add_edge(input_node, output_node, weight=float(inputs.mean()))

        for node in list(self.hypergraph.nodes())[:5]:
            if node not in (input_node, output_node):
                self.hypergraph.add_edge(input_node, node, weight=float(inputs.mean()))

        for node in list(self.hypergraph.nodes())[-5:]:
            if node not in (input_node, output_node):
                self.hypergraph.add_edge(output_node, node, weight=float(output.mean()))


class UAV:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.initial_height = z
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.target_height = z
        self.heading = 0  # Heading in radians (0 is facing right, increases counterclockwise)
        self.landed = False

    def move(self):
        if self.landed:
            return

        self.x += self.vel_x * VELOCITY_FACTOR
        self.y += self.vel_y * VELOCITY_FACTOR
        
        # Adjust height towards target height
        height_difference = self.target_height - self.z
        self.vel_z += 0.1 * height_difference  # Proportional control
        self.vel_z = max(min(self.vel_z, 1), -1)  # Limit vertical velocity
        
        self.z += self.vel_z * VELOCITY_FACTOR

        #  Keep UAV within bounds (wall avoidance)
        margin = 20  # Distance from wall to start avoidance
        if self.x < margin:
            self.heading = 0  # Face right
        elif self.x > WIN_WIDTH - margin:
            self.heading = math.pi  # Face left
        if self.y < margin:
            self.heading = math.pi / 2  # Face up
        elif self.y > WIN_HEIGHT - margin:
            self.heading = 3 * math.pi / 2  # Face down

        self.x = max(min(self.x, WIN_WIDTH), 0)
        self.y = max(min(self.y, WIN_HEIGHT), 0)
        self.z = max(min(self.z, ROOM_HEIGHT), 0)  # Use ROOM_HEIGHT instead of 100

    def land(self):
        self.landed = True
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.z = 0  # Set height to ground level

    def detect_obstacles(self, obstacles, detection_range=150):
        nearby_obstacles = []
        for obstacle in obstacles:
            distance = ((self.x - obstacle.x)**2 + (self.y - obstacle.y)**2 + (self.z - obstacle.z)**2)**0.5
            if distance < detection_range:
                nearby_obstacles.append((obstacle, distance))
        return sorted(nearby_obstacles, key=lambda x: x[1])[:4]  # Return the 4 nearest obstacles

    def adjust_movement(self, obstacles, target):
        if self.landed:
            return

        # Obstacle avoidance
        nearby_obstacles = self.detect_obstacles(obstacles)
        repulsion_x, repulsion_y, repulsion_z = 0, 0, 0
        for obstacle, distance in nearby_obstacles:
            if distance < 50:  # If very close to an obstacle
                repulsion_force = 5 / (distance + 1)  # Increased force, avoid division by zero
                angle_to_obstacle = math.atan2(obstacle.y - self.y, obstacle.x - self.x)
                repulsion_x += repulsion_force * math.cos(angle_to_obstacle + math.pi)
                repulsion_y += repulsion_force * math.sin(angle_to_obstacle + math.pi)
                
                # Improve z-axis avoidance
                z_difference = self.z - obstacle.z
                if abs(z_difference) < 30:  # If close in z-axis
                    repulsion_z += repulsion_force * (1 if z_difference > 0 else -1)

        # Apply repulsion to heading and vertical velocity
        if repulsion_x != 0 or repulsion_y != 0:
            repulsion_angle = math.atan2(repulsion_y, repulsion_x)
            self.heading = (0.7 * self.heading + 0.3 * repulsion_angle) % (2 * math.pi)
        
        # Adjust vertical velocity based on repulsion
        self.vel_z += 0.2 * repulsion_z  # Increased influence of z-axis repulsion

        # Target seeking
        distance_to_target = math.sqrt((target[0] - self.x)**2 + (target[1] - self.y)**2 + (target[2] - self.z)**2)
        
        if distance_to_target < 10:  # If very close to target
            self.land()
            return

        angle_to_target = math.atan2(target[1] - self.y, target[0] - self.x)
        angle_diff = angle_to_target - self.heading
        
        # Normalize angle difference to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Adjust heading towards target
        self.heading += 0.1 * angle_diff

        # Normalize heading to [0, 2pi]
        self.heading = self.heading % (2 * math.pi)

        # Set speed based on distance to target
        speed = min(distance_to_target / 100, 1)  # Limit max speed

        # Update velocities
        self.vel_x = speed * math.cos(self.heading)
        self.vel_y = speed * math.sin(self.heading)
        
        # Adjust target height considering obstacles
        self.target_height = target[2]
        for obstacle, distance in nearby_obstacles:
            if abs(self.z - obstacle.z) < 30 and distance < 50:
                self.target_height = max(self.target_height, obstacle.z + 40)  # Aim to fly at least 40 units above obstacles

        # Adjust vertical velocity towards target height
        height_difference = self.target_height - self.z
        self.vel_z += 0.1 * height_difference  # Proportional control

        # Clamp velocities
        self.vel_x = max(min(self.vel_x, 1), -1)
        self.vel_y = max(min(self.vel_y, 1), -1)
        self.vel_z = max(min(self.vel_z, 1), -1)

    def draw(self, win):
        screen_x = int(self.x)
        screen_y = int(WIN_HEIGHT - self.y - self.z)
        color = (0, 0, 255) if self.landed else (255, 0, 0)  # Blue when landed, Red when flying
        pygame.draw.circle(win, color, (screen_x, screen_y), 10)
        
        if not self.landed:
            # Draw heading indicator
            end_x = screen_x + 20 * math.cos(self.heading)
            end_y = screen_y - 20 * math.sin(self.heading)
            pygame.draw.line(win, (0, 0, 255), (screen_x, screen_y), (int(end_x), int(end_y)), 3)


class Obstacle:
    def __init__(self, x, y, z, radius):
        self.x = max(radius, min(x, ROOM_WIDTH - radius))
        self.y = max(radius, min(y, ROOM_LENGTH - radius))
        self.z = max(radius, min(z, ROOM_HEIGHT - radius))
        self.radius = radius

    def draw(self, win):
        screen_x = int(self.x)
        screen_y = int(WIN_HEIGHT - self.y - self.z)
        pygame.draw.circle(win, (0, 255, 0), (screen_x, screen_y), self.radius)
        
        # Display obstacle position and height
        font = pygame.font.Font(None, 24)
        text = font.render(f"({self.x}, {self.y}, {self.z})", True, (0, 0, 0))
        win.blit(text, (screen_x + self.radius, screen_y - self.radius))

    def collide(self, uav):
        distance = ((self.x - uav.x)**2 + (self.y - uav.y)**2 + (self.z - uav.z)**2)**0.5
        return distance < self.radius

class HypergraphVisualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

    def update(self, graph):
        self.surface.fill((255, 255, 255))
        pos = nx.spring_layout(graph)
        for node, (x, y) in pos.items():
            color = (0, 0, 255)  # Blue for nodes
            pygame.draw.circle(self.surface, color, 
                               (int(x * self.width), int(y * self.height)), 5)
            
            font = pygame.font.Font(None, 20)
            label = font.render(str(node), True, (0, 0, 0))
            self.surface.blit(label, (int(x * self.width) + 5, int(y * self.height) + 5))

        for edge in graph.edges():
            start = pos[edge[0]]
            end = pos[edge[1]]
            pygame.draw.line(self.surface, (255, 0, 0),  
                             (int(start[0] * self.width), int(start[1] * self.height)),
                             (int(end[0] * self.width), int(end[1] * self.height)))
        return self.surface

def get_nearby_obstacles_info(self, obstacles):
    nearby = self.detect_obstacles(obstacles)
    return [f"Obs {i+1}: ({o[0].x:.0f}, {o[0].y:.0f}, {o[0].z:.0f}) Dist: {o[1]:.0f}" for i, o in enumerate(nearby)]


def draw_window(win, uav_or_uavs, obstacles, score, gen, hypergraph_surface):
    # Define dimensions
    MAIN_AREA_WIDTH = int(WIN_WIDTH * 0.8)
    INFO_PANEL_WIDTH = WIN_WIDTH - MAIN_AREA_WIDTH
    
    # Clear the window with dark background
    win.fill((20, 20, 30))  # Dark blue-gray background

    # Draw main simulation area
    main_surface = pygame.Surface((MAIN_AREA_WIDTH, WIN_HEIGHT))
    main_surface.fill((30, 30, 40))  # Slightly lighter dark background

    # Draw grid
    grid_color = (50, 50, 60)
    for x in range(0, MAIN_AREA_WIDTH, 50):
        pygame.draw.line(main_surface, grid_color, (x, 0), (x, WIN_HEIGHT))
    for y in range(0, WIN_HEIGHT, 50):
        pygame.draw.line(main_surface, grid_color, (0, y), (MAIN_AREA_WIDTH, y))

    # Draw room boundaries
    room_color = (100, 100, 110)
    scaled_room_width = int(ROOM_WIDTH * MAIN_AREA_WIDTH / WIN_WIDTH)
    pygame.draw.rect(main_surface, room_color, (0, 0, scaled_room_width, ROOM_LENGTH), 2)
    

    # Draw home position
    scaled_home_x = int(HOME_POS[0] * scaled_room_width / ROOM_WIDTH)
    scaled_home_y = int(HOME_POS[1] * ROOM_LENGTH / ROOM_LENGTH)
    pygame.draw.circle(main_surface, (0, 100, 255), (scaled_home_x, ROOM_LENGTH - scaled_home_y), 15)
    
    # Draw target position
    scaled_target_x = int(TARGET_POS[0] * scaled_room_width / ROOM_WIDTH)
    scaled_target_y = int(TARGET_POS[1] * ROOM_LENGTH / ROOM_LENGTH)
    pygame.draw.circle(main_surface, (255, 100, 255), (scaled_target_x, ROOM_LENGTH - scaled_target_y), 15)
    
    # Draw obstacles
    font = pygame.font.Font(None, 20)
    for obstacle in obstacles:
        scaled_x = int(obstacle.x * scaled_room_width / ROOM_WIDTH)
        scaled_y = int(obstacle.y * ROOM_LENGTH / ROOM_LENGTH)
        scaled_z = int(obstacle.z * ROOM_LENGTH / ROOM_HEIGHT)
        pygame.draw.circle(main_surface, (0, 255, 0), (scaled_x, ROOM_LENGTH - scaled_y), obstacle.radius)
        
        # Display obstacle position and height
        text = font.render(f"({obstacle.x:.0f}, {obstacle.y:.0f}, {obstacle.z:.0f})", True, (200, 200, 200))
        main_surface.blit(text, (scaled_x + obstacle.radius, ROOM_LENGTH - scaled_y - obstacle.radius))

    # Draw UAV(s)
    if isinstance(uav_or_uavs, list):
        for uav in uav_or_uavs:
            scaled_x = int(uav.x * scaled_room_width / ROOM_WIDTH)
            scaled_y = int(uav.y * ROOM_LENGTH / ROOM_LENGTH)
            scaled_z = int(uav.z * ROOM_LENGTH / ROOM_HEIGHT)
            color = (0, 0, 255) if uav.landed else (255, 0, 0)
            pygame.draw.circle(main_surface, color, (scaled_x, ROOM_LENGTH - scaled_y), 10)
            
            if not uav.landed:
                # Draw heading indicator
                end_x = scaled_x + 20 * math.cos(uav.heading)
                end_y = (ROOM_LENGTH - scaled_y) - 20 * math.sin(uav.heading)
                pygame.draw.line(main_surface, (0, 0, 255), (scaled_x, ROOM_LENGTH - scaled_y), (int(end_x), int(end_y)), 3)
    else:
        uav = uav_or_uavs
        scaled_x = int(uav.x * scaled_room_width / ROOM_WIDTH)
        scaled_y = int(uav.y * ROOM_LENGTH / ROOM_LENGTH)
        scaled_z = int(uav.z * ROOM_LENGTH / ROOM_HEIGHT)
        color = (0, 0, 255) if uav.landed else (255, 0, 0)
        pygame.draw.circle(main_surface, color, (scaled_x, ROOM_LENGTH - scaled_y), 10)
        
        if not uav.landed:
            # Draw heading indicator
            end_x = scaled_x + 20 * math.cos(uav.heading)
            end_y = (ROOM_LENGTH - scaled_y) - 20 * math.sin(uav.heading)
            pygame.draw.line(main_surface, (0, 0, 255), (scaled_x, ROOM_LENGTH - scaled_y), (int(end_x), int(end_y)), 3)
    
    # Draw main surface on window
    win.blit(main_surface, (0, 0))

    # Draw info panel
    info_panel = pygame.Surface((INFO_PANEL_WIDTH, WIN_HEIGHT))
    info_panel.fill((40, 40, 50))  # Dark background for info panel

    # Render text for info panel
    font = pygame.font.Font(None, 24)
    y_offset = 10

    texts = [
        f"Generation: {gen}",
        f"Score: {score}",
    ]

    if isinstance(uav_or_uavs, list):
        for i, uav in enumerate(uav_or_uavs):
            texts.extend([
                f"UAV {i} Position:",
                f"X: {uav.x:.1f}",
                f"Y: {uav.y:.1f}",
                f"Z: {uav.z:.1f}",
                f"Velocity:",
                f"X: {uav.vel_x:.2f}",
                f"Y: {uav.vel_y:.2f}",
                f"Z: {uav.vel_z:.2f}",
                f"Land Status: {'Landed' if uav.landed else 'Flying'}"
            ])
            texts.append("Nearby Obstacles:")
            texts.extend(get_nearby_obstacles_info(uav, obstacles))

    for text in texts:
        text_surface = font.render(text, True, (200, 200, 200))
        info_panel.blit(text_surface, (10, y_offset))
        y_offset += 30

    # Draw info panel on window
    win.blit(info_panel, (MAIN_AREA_WIDTH, 0))

    # Draw hypergraph if enabled
    if SHOW_HYPERGRAPH and hypergraph_surface is not None:
        scaled_surface = pygame.transform.scale(hypergraph_surface, (INFO_PANEL_WIDTH, INFO_PANEL_WIDTH))
        win.blit(scaled_surface, (MAIN_AREA_WIDTH, WIN_HEIGHT - INFO_PANEL_WIDTH))

    pygame.display.update()

def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    uavs = []

    vae = VAE(CA_SIZE, LATENT_DIM)

    for _, g in genomes:
        net = NEATCAHypergraphSNN(g, config, vae)
        nets.append(net)
        uavs.append(UAV(HOME_POS[0], HOME_POS[1], HOME_POS[2]))
        g.fitness = 0
        ge.append(g)

    obstacles = [Obstacle(random.randint(0, ROOM_WIDTH), 
                      random.randint(0, ROOM_LENGTH), 
                      random.randint(0, ROOM_HEIGHT), 
                      random.randint(10, 30)) for _ in range(NUM_OBSTACLES)]
    score = 0

    clock = pygame.time.Clock()

    if SHOW_HYPERGRAPH:
        hypergraph = nx.Graph()
        hypergraph_visualizer = HypergraphVisualizer(HYPERGRAPH_WIDTH, HYPERGRAPH_HEIGHT)
    else:
        hypergraph = None
        hypergraph_visualizer = None

    run = True
    while run and len(uavs) > 0:
        clock.tick(30)
        score += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return

        for x, uav in enumerate(uavs):
            if uav.landed:
                continue

            ge[x].fitness += 0.1

            nearby_obstacles = uav.detect_obstacles(obstacles)

            inputs = [
                uav.x / ROOM_WIDTH,
                uav.y / ROOM_LENGTH,
                uav.z / ROOM_HEIGHT,
                uav.vel_x / 5,
                uav.vel_y / 5,
                uav.vel_z / 5,
                uav.heading / (2 * math.pi),
                (TARGET_POS[0] - uav.x) / ROOM_WIDTH,
                (TARGET_POS[1] - uav.y) / ROOM_LENGTH,
                (TARGET_POS[2] - uav.z) / ROOM_HEIGHT
            ]

            for i in range(3):
                if i < len(nearby_obstacles):
                    obstacle, distance = nearby_obstacles[i]
                    inputs.extend([
                        (obstacle.x - uav.x) / ROOM_WIDTH,
                        (obstacle.y - uav.y) / ROOM_LENGTH,
                        (obstacle.z - uav.z) / ROOM_HEIGHT,
                        distance / ((ROOM_WIDTH**2 + ROOM_LENGTH**2 + ROOM_HEIGHT**2)**0.5)
                    ])
                else:
                    inputs.extend([0, 0, 0, 1])

            output = nets[x].activate(inputs)

            new_heading = output[0] * 2 * math.pi
            new_speed = max(min(output[1], 1), 0)
            new_vel_z = max(min(output[2] * 2 - 1, 1), -1)

            smoothing_factor = 0.1
            uav.heading = uav.heading * (1 - smoothing_factor) + new_heading * smoothing_factor
            uav.vel_x = new_speed * math.cos(uav.heading)
            uav.vel_y = new_speed * math.sin(uav.heading)
            uav.vel_z = uav.vel_z * (1 - smoothing_factor) + new_vel_z * smoothing_factor

            uav.adjust_movement(obstacles, TARGET_POS)
            uav.move()

            if SHOW_HYPERGRAPH:
                hypergraph.add_node(x, label=f'UAV {x}')
                for i, (obstacle, distance) in enumerate(nearby_obstacles):
                    hypergraph.add_node(f'obstacle{i}', label=f'Obstacle {i}')
                    hypergraph.add_edge(x, f'obstacle{i}', weight=1/distance)

            if (uav.x < 0 or uav.x > ROOM_WIDTH or uav.y < 0 or uav.y > ROOM_LENGTH or 
                uav.z < 0 or uav.z > ROOM_HEIGHT or any(obstacle.collide(uav) for obstacle in obstacles)):
                ge[x].fitness -= 10
                uavs.pop(x)
                nets.pop(x)
                ge.pop(x)
            else:
                ge[x].fitness += 0.2
                distance_to_target = math.sqrt((TARGET_POS[0] - uav.x)**2 + (TARGET_POS[1] - uav.y)**2 + (TARGET_POS[2] - uav.z)**2)
                ge[x].fitness += 1 / (distance_to_target + 1)  # Reward getting closer to the target

        # Check if all UAVs have landed
        if all(uav.landed for uav in uavs):
            run = False

        if SHOW_HYPERGRAPH:
            hypergraph_surface = hypergraph_visualizer.update(hypergraph)
        else:
            hypergraph_surface = None

        draw_window(WIN, uavs, obstacles, score, GEN, hypergraph_surface)

    print(f"Generation {GEN} completed. Best score: {score}")

    if ge:
        best_genome = max(ge, key=lambda g: g.fitness)
        with open(f"best_genome_gen_{GEN}.pickle", "wb") as f:
            pickle.dump(best_genome, f)


def run_best_genome(config_path, genome_path):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the best genome.
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Create the network for the genome.
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Initialize Pygame
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Best UAV Simulation")
    clock = pygame.time.Clock()

    # Create UAV and obstacles
    uav = UAV(WIN_WIDTH // 2, WIN_HEIGHT // 2, 50)
    obstacles = [Obstacle(random.randint(0, WIN_WIDTH), random.randint(0, WIN_HEIGHT), 
                          random.randint(0, 100), random.randint(10, 30)) for _ in range(10)]

    running = True
    gen = 0
    score = 0
    while running:
        clock.tick(30)  # Limit to 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get inputs for the network
        nearby_obstacles = uav.detect_obstacles(obstacles)
        inputs = [
            uav.x / WIN_WIDTH,
            uav.y / WIN_HEIGHT,
            uav.z / 100,
            (uav.z - uav.initial_height) / 100,
            uav.vel_x / 5,
            uav.vel_y / 5,
            uav.vel_z / 5
        ]

        # Add information about nearby obstacles
        for i in range(4):
            if i < len(nearby_obstacles):
                obstacle, distance = nearby_obstacles[i]
                inputs.extend([
                    (obstacle.x - uav.x) / WIN_WIDTH,
                    (obstacle.y - uav.y) / WIN_HEIGHT,
                    (obstacle.z - uav.z) / 100,
                    distance / 150
                ])
            else:
                inputs.extend([0, 0, 0, 1])

        # Get output from the network
        output = net.activate(inputs)

        # Update UAV velocities
        new_vel_x = max(min(output[0] * 2 - 1, 1), -1)
        new_vel_y = max(min(output[1] * 2 - 1, 1), -1)
        new_vel_z = max(min(output[2] * 2 - 1, 1), -1)

        # Smooth the velocities
        smoothing_factor = 0.1
        uav.vel_x = uav.vel_x * (1 - smoothing_factor) + new_vel_x * smoothing_factor
        uav.vel_y = uav.vel_y * (1 - smoothing_factor) + new_vel_y * smoothing_factor
        uav.vel_z = uav.vel_z * (1 - smoothing_factor) + new_vel_z * smoothing_factor

        uav.adjust_movement(obstacles, TARGET_POS)
        uav.move()

        # Check for collisions
        for obstacle in obstacles:
            if obstacle.collide(uav):
                print("UAV collided with an obstacle!")
                running = False
                break

        # Update score (you can define your own scoring method)
        score += 1

        # Draw everything
        draw_window(win, uav, obstacles, gen, score, None)

    pygame.quit()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))

    with open("best_genome.pickle", "wb") as f:
        pickle.dump(winner, f)

    print("\nTraining complete. Press any key to watch the best genome in action...")
    input()

    run_best_genome(config_path, "best_genome.pickle")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-uav.ini')
    run(config_path)
