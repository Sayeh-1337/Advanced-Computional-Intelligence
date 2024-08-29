import pygame
import neat
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
from collections import deque

# Initialize Pygame
pygame.init()

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 100
PADDLE_SPEED = 6
BALL_SIZE = 15
BALL_SPEED = 7
NEAT_GENERATIONS = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# CA constants
CA_SIZE = 16
CA_RULES = [110, 30, 90, 184, 57, 54, 18, 45]  # Expanded rule set

# Pygame setup
screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption("Enhanced NEAT-CA Pong")
clock = pygame.time.Clock()

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move(self, dy):
        self.rect.y = max(0, min(GAME_HEIGHT - PADDLE_HEIGHT, self.rect.y + dy))

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(GAME_WIDTH // 2 - BALL_SIZE // 2, GAME_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.dx = -BALL_SPEED
        self.dy = random.uniform(-3, 3)

    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        if self.rect.top <= 0 or self.rect.bottom >= GAME_HEIGHT:
            self.dy *= -1

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

def cellular_automaton_step(states):
    rule = random.choice(CA_RULES)
    new_states = np.zeros_like(states)
    for i in range(CA_SIZE):
        neighborhood = (int(states[(i-1)%CA_SIZE] > 0.5) << 2) | (int(states[i] > 0.5) << 1) | int(states[(i+1)%CA_SIZE] > 0.5)
        new_states[i] = 1 if (rule & (1 << neighborhood)) else 0
    return new_states

def eval_genomes(genomes, config):
    global best_genome, best_fitness

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        paddle = Paddle(10, GAME_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        ball = Ball()
        score = 0
        node_states = np.random.random(CA_SIZE)

        for _ in range(2000):  # Increased frames for better evaluation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Enhanced inputs
            inputs = [
                paddle.rect.centery / GAME_HEIGHT,
                ball.rect.centerx / GAME_WIDTH,
                ball.rect.centery / GAME_HEIGHT,
                ball.dx / BALL_SPEED,
                ball.dy / BALL_SPEED,
                (ball.rect.centery - paddle.rect.centery) / GAME_HEIGHT,
                (GAME_WIDTH - ball.rect.centerx) / GAME_WIDTH
            ] + [int(state > 0.5) for state in node_states]

            output = net.activate(inputs)[0]

            paddle_target = output * GAME_HEIGHT
            paddle_move = np.clip(paddle_target - paddle.rect.centery, -PADDLE_SPEED, PADDLE_SPEED)
            paddle.move(paddle_move)

            ball.move()

            if ball.rect.left <= 0:
                if ball.rect.centery >= paddle.rect.top and ball.rect.centery <= paddle.rect.bottom:
                    ball.dx *= -1
                    ball.dy += random.uniform(-1, 1)
                    score += 1
                else:
                    break

            if ball.rect.right >= GAME_WIDTH:
                ball.dx *= -1

            node_states = cellular_automaton_step(node_states)

        # Enhanced fitness function
        fitness = score * 10 + _ / 100 + (score ** 2)  # Reward higher scores more
        genome.fitness = fitness

        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome

def run_neat(config):
    global best_genome, best_fitness
    best_genome = None
    best_fitness = 0

    # Create population and add reporters
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(eval_genomes, NEAT_GENERATIONS)

    # Save the winner genome
    with open('winner_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    print(f"Best fitness achieved: {best_fitness}")
    print("Winner genome saved to 'winner_genome.pkl'")

    return stats

def plot_stats(stats):
    gen = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    plt.figure(figsize=(12, 8))
    plt.plot(gen, best_fitness, 'r-', label="Best Fitness")
    plt.plot(gen, avg_fitness, 'b-', label="Average Fitness")
    plt.fill_between(gen, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, facecolor="blue", alpha=0.2)
    plt.title("NEAT-CA Pong Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.savefig("neat_ca_pong_stats.png")
    plt.close()

def visualize_winner(config):
    # Load the winner genome
    with open('winner_genome.pkl', 'rb') as f:
        winner_genome = pickle.load(f)

    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    paddle = Paddle(10, GAME_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()
    score = 0
    node_states = np.random.random(CA_SIZE)
    
    # For calculating average score
    scores = deque(maxlen=100)
    frames = 0

    font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        inputs = [
            paddle.rect.centery / GAME_HEIGHT,
            ball.rect.centerx / GAME_WIDTH,
            ball.rect.centery / GAME_HEIGHT,
            ball.dx / BALL_SPEED,
            ball.dy / BALL_SPEED,
            (ball.rect.centery - paddle.rect.centery) / GAME_HEIGHT,
            (GAME_WIDTH - ball.rect.centerx) / GAME_WIDTH
        ] + [int(state > 0.5) for state in node_states]

        output = winner_net.activate(inputs)[0]

        paddle_target = output * GAME_HEIGHT
        paddle_move = np.clip(paddle_target - paddle.rect.centery, -PADDLE_SPEED, PADDLE_SPEED)
        paddle.move(paddle_move)

        ball.move()

        if ball.rect.left <= 0:
            if ball.rect.centery >= paddle.rect.top and ball.rect.centery <= paddle.rect.bottom:
                ball.dx *= -1
                ball.dy += random.uniform(-1, 1)
                score += 1
            else:
                scores.append(score)
                ball.reset()
                score = 0

        if ball.rect.right >= GAME_WIDTH:
            ball.dx *= -1

        node_states = cellular_automaton_step(node_states)

        screen.fill(BLACK)
        paddle.draw()
        ball.draw()

        # Display score and other info
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        avg_score = sum(scores) / len(scores) if scores else 0
        avg_score_text = font.render(f"Avg Score: {avg_score:.2f}", True, WHITE)
        screen.blit(avg_score_text, (10, 50))

        frames_text = font.render(f"Frames: {frames}", True, WHITE)
        screen.blit(frames_text, (10, 90))

        # Visualize CA states
        for i, state in enumerate(node_states):
            color = WHITE if state > 0.5 else RED
            pygame.draw.rect(screen, color, (GAME_WIDTH - 20, i * 20, 10, 10))

        pygame.display.flip()
        clock.tick(60)
        frames += 1

    pygame.quit()

if __name__ == "__main__":
    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pong_neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Run NEAT and get stats
    stats = run_neat(config)

    # Plot and save statistics
    plot_stats(stats)

    # Visualize the performance of the best genome
    visualize_winner(config)