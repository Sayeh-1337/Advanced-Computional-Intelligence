import pygame
import neat
import random
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import base64
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle

# Initialize Pygame
pygame.init()

# Game Constants
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

# CA and VAE Constants
CA_SIZE = 50
LATENT_DIM = 10

# Hypergraph visualization constants
HYPERGRAPH_WIDTH = 300
HYPERGRAPH_HEIGHT = 300
show_hypergraph = True  # Replace the constant show_hypergraph with this variable

# Define GEN as a global variable
GEN = 0

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class NEATCAHypergraph:
    def __init__(self, genome, config, vae):
        self.neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.vae = vae
        self.hypergraph = nx.Graph()
        self.ca_state = np.random.random(CA_SIZE)

    def activate(self, inputs):
        ca_state_tensor = torch.FloatTensor(self.ca_state).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.vae.encode(ca_state_tensor)
        latent_ca = mu.squeeze().numpy()

        # Use only the original inputs for the NEAT network
        neat_output = self.neat_network.activate(inputs)

        # Ensure neat_output is the correct size (LATENT_DIM)
        if len(neat_output) != LATENT_DIM:
            neat_output = np.pad(neat_output, (0, LATENT_DIM - len(neat_output)), 'constant')

        neat_output_tensor = torch.FloatTensor(neat_output).unsqueeze(0)
        with torch.no_grad():
            new_ca_state = self.vae.decode(neat_output_tensor)
        self.ca_state = new_ca_state.squeeze().numpy()

        self.update_hypergraph(inputs, latent_ca, neat_output)

        return neat_output[0]

    def update_hypergraph(self, inputs, latent_ca, output):
        input_node = self.hypergraph.number_of_nodes()
        ca_node = input_node + 1
        output_node = ca_node + 1

        self.hypergraph.add_node(input_node, type="input")
        self.hypergraph.add_node(ca_node, type="ca")
        self.hypergraph.add_node(output_node, type="output")

        self.hypergraph.add_edge(input_node, output_node, weight=float(np.mean(inputs)))
        self.hypergraph.add_edge(ca_node, output_node, weight=float(np.mean(latent_ca)))

class Bird:
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False

class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

class HypergraphVisualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)

    def update(self, hypergraph, node_states):
        self.ax.clear()
        pos = nx.spring_layout(hypergraph)
        
        # Draw nodes
        nx.draw_networkx_nodes(hypergraph, pos, ax=self.ax, 
                               node_color=[node_states.get(n, 0.5) for n in hypergraph.nodes()],
                               cmap='coolwarm', node_size=300)
        
        # Draw edges
        nx.draw_networkx_edges(hypergraph, pos, ax=self.ax)
        
        # Draw labels
        nx.draw_networkx_labels(hypergraph, pos, ax=self.ax, font_size=8)
        
        self.ax.set_title('Hypergraph Visualization')
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()
        return pygame.image.fromstring(raw_data, size, "RGB")

def update_hypergraph(hypergraph, bird, pipes):
    node_states = {}
    
    # Update node states
    for node in hypergraph.nodes():
        if isinstance(node, int):
            node_states[node] = bird.y / FLOOR if bird else 0.5
        elif isinstance(node, str) and node.startswith('pipe'):
            pipe_idx = int(node[4:])
            if pipe_idx < len(pipes):
                node_states[node] = pipes[pipe_idx].x / WIN_WIDTH
            else:
                node_states[node] = 0.5
    
    # Update edges (this is a simple example, adjust according to your needs)
    hypergraph.clear_edges()
    if bird:
        for pipe_node in [node for node in hypergraph.nodes() if isinstance(node, str) and node.startswith('pipe')]:
            hypergraph.add_edge(0, pipe_node, weight=abs(bird.y - pipes[int(pipe_node[4:])].height) if int(pipe_node[4:]) < len(pipes) else 1)
    
    return node_states

def draw_window(win, birds, pipes, base, score, gen, hypergraph_surface):
    global show_hypergraph
    win.blit(bg_img, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    # Draw hypergraph only if show_hypergraph is True
    if show_hypergraph and hypergraph_surface is not None:
        win.blit(hypergraph_surface, (WIN_WIDTH - HYPERGRAPH_WIDTH, WIN_HEIGHT - HYPERGRAPH_HEIGHT))

    pygame.display.update()

def eval_genomes(genomes, config):
    global GEN, show_hypergraph
    GEN += 1

    nets = []
    ge = []
    birds = []
    
    # Initialize VAE
    vae = VAE(CA_SIZE, LATENT_DIM)
    
    for _, g in genomes:
        net = NEATCAHypergraph(g, config, vae)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(FLOOR)
    pipes = [Pipe(WIN_WIDTH)]
    score = 0

    clock = pygame.time.Clock()

    # Initialize hypergraph only if it will be shown
    if show_hypergraph:
        hypergraph = nx.Graph()
        hypergraph.add_node(0, label='Bird')  # Bird node
        for i in range(3):
            hypergraph.add_node(f'pipe{i}', label=f'Pipe {i}')
        hypergraph_visualizer = HypergraphVisualizer(HYPERGRAPH_WIDTH, HYPERGRAPH_HEIGHT)
    else:
        hypergraph = None
        hypergraph_visualizer = None

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    show_hypergraph = not show_hypergraph
                    print(f"Hypergraph visibility toggled: {'On' if show_hypergraph else 'Off'}")

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            # Prepare inputs for the network
            inputs = (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))
            
            # Activate the NEAT-CA Hypergraph
            output = nets[x].activate(inputs)

            if output > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= FLOOR or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        # Update and visualize hypergraph only if show_hypergraph is True
        if show_hypergraph:
            node_states = update_hypergraph(hypergraph, birds[0] if birds else None, pipes)
            hypergraph_surface = hypergraph_visualizer.update(hypergraph, node_states)
        else:
            hypergraph_surface = None

        draw_window(WIN, birds, pipes, base, score, GEN, hypergraph_surface)

        if score > 50:
            break

    # After the game loop ends
    print(f"Generation {GEN} completed. Best score: {score}")
    
    # Save the best performing network
    if len(ge) > 0:
        best_genome = max(ge, key=lambda g: g.fitness)
        with open(f"best_genome_gen_{GEN}.pickle", "wb") as f:
            pickle.dump(best_genome, f)
        
def run_best_network(genome, config):
    global show_hypergraph
    # Initialize Pygame
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Best Network")

    # Create the bird and other game objects
    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(WIN_WIDTH)]
    
    # Initialize VAE and NEAT-CA Hypergraph
    vae = VAE(CA_SIZE, LATENT_DIM)
    net = NEATCAHypergraph(genome, config, vae)
    
    # Initialize hypergraph only if it will be shown
    if show_hypergraph:
        hypergraph = nx.Graph()
        hypergraph.add_node(0, label='Bird')  # Bird node
        for i in range(3):
            hypergraph.add_node(f'pipe{i}', label=f'Pipe {i}')
        hypergraph_visualizer = HypergraphVisualizer(HYPERGRAPH_WIDTH, HYPERGRAPH_HEIGHT)
    else:
        hypergraph = None
        hypergraph_visualizer = None


    clock = pygame.time.Clock()
    score = 0
    run = True
    gen= "best_gen"

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    show_hypergraph = not show_hypergraph
                    print(f"Hypergraph visibility toggled: {'On' if show_hypergraph else 'Off'}")

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        output = net.activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

        if output > 0.5:
            bird.jump()

        bird.move()
        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                run = False

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
            run = False

        # Update and visualize hypergraph only if show_hypergraph is True
        if show_hypergraph:
            node_states = update_hypergraph(hypergraph, bird, pipes)
            hypergraph_surface = hypergraph_visualizer.update(hypergraph, node_states)
        else:
            hypergraph_surface = None

        # Update the draw_window call
        draw_window(win, [bird], pipes, base, score, gen, hypergraph_surface)

    print(f"Final Score: {score}")

def visualize_winner(winner_net):
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(winner_net.hypergraph)
    nx.draw(winner_net.hypergraph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(winner_net.hypergraph, 'weight')
    nx.draw_networkx_edge_labels(winner_net.hypergraph, pos, edge_labels=edge_labels)
    
    plt.title("Winner's Hypergraph")
    plt.axis('off')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <title>Winner's Hypergraph</title>
    </head>
    <body>
        <h1>Winner's Hypergraph</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Winner's Hypergraph">
    </body>
    </html>
    """

    # Save HTML file
    with open('winner_hypergraph.html', 'w') as f:
        f.write(html_content)

    print("Graph saved as 'winner_hypergraph.html'")
    plt.close()



def run(config_file):
    global show_hypergraph

    def handle_events():
        global show_hypergraph
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:  # Press 'h' to toggle hypergraph
                    show_hypergraph = not show_hypergraph

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))

    # Save statistics
    stats.save_species_count()
    stats.save_species_fitness()

    # Print final statistics
    print("\nFinal Statistics")
    print("Best Fitness:", stats.best_genome().fitness)
    print("Species Count:", len(stats.get_species_sizes()))
    species_sizes = stats.get_species_sizes()
    print("Type of species_sizes:", type(species_sizes))
    print("Content of species_sizes:", species_sizes)

    if isinstance(species_sizes, list):
        print("species_sizes is a list. Attempting to sum...")
        try:
            total = sum(species_sizes)
            print("Total Population:", total)
        except TypeError as e:
            print("Error when summing species_sizes:", e)
            print("Types of elements in species_sizes:")
            for item in species_sizes:
                print(type(item))
    elif isinstance(species_sizes, dict):
        print("species_sizes is a dict. Summing values...")
        print("Total Population:", sum(species_sizes.values()))
    else:
        print("species_sizes is neither a list nor a dict.")
        print("Unable to calculate total population.")
        print("Unexpected type for species_sizes:", type(species_sizes))
        print("Content of species_sizes:", species_sizes)
        print("Average Fitness:", stats.get_fitness_mean())
        print("Fitness Stdev:", stats.get_fitness_stdev())

    # Visualize the winner's neural network and hypergraph
    winner_net = NEATCAHypergraph(winner, config, VAE(CA_SIZE, LATENT_DIM))
    
    # Call the visualization function
    visualize_winner(winner_net)
    # Run the best network
    run_best_network(winner, config)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.ini')
    run(config_path)