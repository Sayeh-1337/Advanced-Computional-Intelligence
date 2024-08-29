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
CA_RULES = [110, 30, 90, 184, 57, 54, 18, 45]

# Pygame setup
screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption("NEAT-CA vs Q-Learning Pong")
clock = pygame.time.Clock()

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move(self, dy):
        self.rect.y = max(0, min(GAME_HEIGHT - PADDLE_HEIGHT, self.rect.y + dy))

    def draw(self, color):
        pygame.draw.rect(screen, color, self.rect)

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(GAME_WIDTH // 2 - BALL_SIZE // 2, GAME_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.dx = random.choice([-BALL_SPEED, BALL_SPEED])
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

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q

def discretize_state(paddle_y, ball_x, ball_y):
    paddle_pos = paddle_y // (GAME_HEIGHT // 10)
    ball_x_pos = ball_x // (GAME_WIDTH // 10)
    ball_y_pos = ball_y // (GAME_HEIGHT // 10)
    return paddle_pos * 100 + ball_x_pos * 10 + ball_y_pos

def train_q_agent():
    q_agent = QLearningAgent(1000, 3)  # 3 actions: up, down, stay
    paddle = Paddle(GAME_WIDTH - PADDLE_WIDTH - 10, GAME_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()

    for episode in range(10000):
        state = discretize_state(paddle.rect.y, ball.rect.x, ball.rect.y)
        total_reward = 0

        while True:
            action = q_agent.get_action(state)
            
            if action == 0:
                paddle.move(-PADDLE_SPEED)
            elif action == 1:
                paddle.move(PADDLE_SPEED)

            ball.move()

            if ball.rect.right >= GAME_WIDTH:
                if ball.rect.centery >= paddle.rect.top and ball.rect.centery <= paddle.rect.bottom:
                    ball.dx *= -1
                    ball.dy += random.uniform(-1, 1)
                    reward = 1
                else:
                    reward = -1
                    ball.reset()
            else:
                reward = 0

            next_state = discretize_state(paddle.rect.y, ball.rect.x, ball.rect.y)
            q_agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if ball.rect.left <= 0:
                ball.reset()
                break

        if episode % 1000 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return q_agent

def load_neat_agent(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    with open('pong_winner_genome.pkl', 'rb') as f:
        winner_genome = pickle.load(f)

    return neat.nn.FeedForwardNetwork.create(winner_genome, config)

def compete(neat_agent, q_agent):
    neat_paddle = Paddle(10, GAME_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    q_paddle = Paddle(GAME_WIDTH - PADDLE_WIDTH - 10, GAME_HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()
    node_states = np.random.random(CA_SIZE)

    neat_score = 0
    q_score = 0
    
    # Improved font setup
    pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 36)
    small_font = pygame.font.Font(pygame.font.get_default_font(), 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # NEAT agent's turn
        neat_inputs = [
            neat_paddle.rect.centery / GAME_HEIGHT,
            ball.rect.centerx / GAME_WIDTH,
            ball.rect.centery / GAME_HEIGHT,
            ball.dx / BALL_SPEED,
            ball.dy / BALL_SPEED,
            (ball.rect.centery - neat_paddle.rect.centery) / GAME_HEIGHT,
            (GAME_WIDTH - ball.rect.centerx) / GAME_WIDTH
        ] + [int(state > 0.5) for state in node_states]

        neat_output = neat_agent.activate(neat_inputs)[0]
        neat_paddle_target = neat_output * GAME_HEIGHT
        neat_paddle_move = np.clip(neat_paddle_target - neat_paddle.rect.centery, -PADDLE_SPEED, PADDLE_SPEED)
        neat_paddle.move(neat_paddle_move)

        # Q-Learning agent's turn
        q_state = discretize_state(q_paddle.rect.y, ball.rect.x, ball.rect.y)
        q_action = q_agent.get_action(q_state)
        
        if q_action == 0:
            q_paddle.move(-PADDLE_SPEED)
        elif q_action == 1:
            q_paddle.move(PADDLE_SPEED)

        ball.move()

        if ball.rect.left <= 0:
            if ball.rect.centery >= neat_paddle.rect.top and ball.rect.centery <= neat_paddle.rect.bottom:
                ball.dx *= -1
                ball.dy += random.uniform(-1, 1)
            else:
                q_score += 1
                ball.reset()

        if ball.rect.right >= GAME_WIDTH:
            if ball.rect.centery >= q_paddle.rect.top and ball.rect.centery <= q_paddle.rect.bottom:
                ball.dx *= -1
                ball.dy += random.uniform(-1, 1)
            else:
                neat_score += 1
                ball.reset()

        node_states = cellular_automaton_step(node_states)

        screen.fill(BLACK)
        neat_paddle.draw(BLUE)
        q_paddle.draw(RED)
        ball.draw()

        # Improved score display
        neat_score_text = font.render(f"NEAT: {neat_score}", True, BLUE)
        q_score_text = font.render(f"Q-Learning: {q_score}", True, RED)
        
        neat_score_rect = neat_score_text.get_rect(topleft=(20, 20))
        q_score_rect = q_score_text.get_rect(topright=(GAME_WIDTH - 20, 20))
        
        # Draw semi-transparent background for score text
        pygame.draw.rect(screen, (*BLACK, 150), neat_score_rect.inflate(20, 10))
        pygame.draw.rect(screen, (*BLACK, 150), q_score_rect.inflate(20, 10))
        
        screen.blit(neat_score_text, neat_score_rect)
        screen.blit(q_score_text, q_score_rect)

        # Add a title
        title_text = small_font.render("NEAT-CA vs Q-Learning Pong", True, WHITE)
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 20))
        pygame.draw.rect(screen, (*BLACK, 150), title_rect.inflate(20, 10))
        screen.blit(title_text, title_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return neat_score, q_score

if __name__ == "__main__":
    # Load NEAT configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pong_neat_config.ini')

    # Train Q-Learning agent
    print("Training Q-Learning agent...")
    q_agent = train_q_agent()

    # Load pre-trained NEAT agent
    print("Loading NEAT agent...")
    neat_agent = load_neat_agent(config_path)

    # Start the competition
    print("Starting the competition...")
    neat_score, q_score = compete(neat_agent, q_agent)

    print(f"Final Score - NEAT: {neat_score}, Q-Learning: {q_score}")