import math
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pygame

# Initialize Pygame
pygame.init()
pygame.time

# Constants
LARGURA = 800  # Width
ALTURA = 600   # Height
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)

# Rocket constants
GRAVIDADE = 0.5
IMPULSO = 1.0
RAPIDEZ_ROTACAO = 0.1
MAX_COMBUSTIVEL = 100
VELOCIDADE_INICIAL = 2

# Neural Network parameters
POPULATION_SIZE = 50
MUTATION_RATE = 0.5
MUTATION_RANGE = 1

# Neural Network architecture
INPUT_SIZE = 7  # Adjusted for the new inputs
HIDDEN_LAYERS = [32, 32]  # Can be adjusted to increase complexity
OUTPUT_SIZE = 3
RAND_X_RANGE = 200
RAND_Y_RANGE = 20
RAND_VX_RANGE = 0
RAND_VY_RANGE = 0
RAND_ANGLE_RANGE = 0.5
SPEED_LIMIT = 30
LANDING_X = LARGURA / 2
# STEPS_PER_FRAME = 50


class NeuralNetwork:
    def __init__(self):
        # Initialize weights for each layer
        self.weights = []
        layer_sizes = [INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_SIZE]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, inputs):
        # Feedforward pass through the network
        activation = inputs
        for weight in self.weights:
            activation = np.dot(activation, weight)
            activation = np.tanh(activation)  # Activation function
        return activation

    def mutate(self, rate=MUTATION_RATE):
        # Apply random mutations to weights
        for i in range(len(self.weights)):
            mask = np.random.random(self.weights[i].shape) < rate
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * MUTATION_RANGE


class Rocket:
    def __init__(self):
        self.reset()

    
    def reset(self):
        # First, determine the initial angle and velocity
        self.angulo = random.uniform(-RAND_ANGLE_RANGE, RAND_ANGLE_RANGE)  # Angle between -45 and 45 degrees
        base_speed = 10 + random.uniform(-RAND_VY_RANGE, RAND_VY_RANGE)
         
        # Calculate velocity components ensuring downward motion
        self.vx = base_speed * math.sin(self.angulo)  # Horizontal component
        self.vy = abs(base_speed * math.cos(self.angulo))  # Vertical component (always positive/downward)
        
        # Calculate starting position by "backtracking" from center
        trajectory_time = 1.0  # How far back in time to start
        
        # Base position (center of screen with random variation)
        target_x = LARGURA /2 + random.uniform(-RAND_X_RANGE, RAND_X_RANGE)
        target_y = ALTURA / 4 + random.uniform(-RAND_Y_RANGE, RAND_Y_RANGE)
        
        # Calculate initial position by moving backwards along the trajectory
        self.x = target_x - (self.vx * trajectory_time)
        self.y = target_y - (self.vy * trajectory_time - 0.5 * GRAVIDADE * trajectory_time * trajectory_time)
        
        # Reset other properties
        self.combustivel = MAX_COMBUSTIVEL
        self.colidiu = False
        self.fitness = 0
        self.success = False
        self.thrusting = False
        self.hit_wall = False

    def apply_thrust(self):
        if self.combustivel > 0:
            force_x = IMPULSO * math.sin(self.angulo)
            force_y = -IMPULSO * math.cos(self.angulo)
            self.vx += force_x
            self.vy += force_y
            self.combustivel -= 1
            self.thrusting = True
        else:
            self.thrusting = False

    def rotate_left(self):
        self.angulo -= RAPIDEZ_ROTACAO

    def rotate_right(self):
        self.angulo += RAPIDEZ_ROTACAO

    def update(self):
        if not self.colidiu and not self.success:
            self.vy += GRAVIDADE
            self.x += self.vx
            self.y += self.vy

            # Boundary checking
            if (self.x < 0 or self.x > LARGURA or self.y > ALTURA or
                (self.y > ALTURA - 10 and (self.x < LANDING_X - 50 or self.x > LANDING_X + 50))):
                if(self.x < 0 or self.x > LARGURA or self.y < 0):
                    self.hit_wall = True
                    self.vy = 10000
                self.colidiu = True

            # Landing success check
            if (self.y > ALTURA - 20 and
                LANDING_X - 50 < self.x < LANDING_X + 50 and
                abs(self.vy) < SPEED_LIMIT and
                abs(self.angulo) < 0.2):
                self.success = True
                self.colidiu = True

    def draw(self, screen):
        if not self.colidiu or self.success:
            # Draw flame if thrusting
            if self.thrusting and self.combustivel > 0:
                flame_points = [
                    (self.x - 3, self.y + 12),
                    (self.x + 3, self.y + 12),
                    (self.x, self.y + 25)
                ]
                rotated_flame = []
                for px, py in flame_points:
                    new_x = (px - self.x) * math.cos(self.angulo) - (py - self.y) * math.sin(self.angulo) + self.x
                    new_y = (px - self.x) * math.sin(self.angulo) + (py - self.y) * math.cos(self.angulo) + self.y
                    rotated_flame.append((new_x, new_y))
                pygame.draw.polygon(screen, VERMELHO, rotated_flame)

            # Draw rocket body
            points = [
                (self.x - 5, self.y - 12),
                (self.x + 5, self.y - 12),
                (self.x + 5, self.y + 12),
                (self.x - 5, self.y + 12)
            ]

            rotated_points = []
            for px, py in points:
                new_x = (px - self.x) * math.cos(self.angulo) - (py - self.y) * math.sin(self.angulo) + self.x
                new_y = (px - self.x) * math.sin(self.angulo) + (py - self.y) * math.cos(self.angulo) + self.y
                rotated_points.append((new_x, new_y))

            pygame.draw.polygon(screen, BRANCO, rotated_points)


class Evolution:
    def __init__(self):
        self.population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
        self.generation = 1
        self.best_fitness = float('-inf')
        self.best_network = None
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.fitness_scores = [0.0] * POPULATION_SIZE
        self.current_best_fitness = float('-inf')
        self.run = 0

    def evaluate_step(self):
        all_done = True
        for i, (rocket, network) in enumerate(zip(self.rockets, self.population)):
            if not rocket.colidiu:
                all_done = False
                landing_pad_x = LANDING_X
                inputs = np.array([
                    rocket.x / LARGURA,
                    rocket.y / ALTURA,
                    rocket.vx / 10,
                    rocket.vy / 10,
                    rocket.angulo / (math.pi / 2),
                    (rocket.x - landing_pad_x) / LARGURA,
                    rocket.combustivel / MAX_COMBUSTIVEL
                ])

                outputs = network.forward(inputs)
                rocket.thrusting = False
                if outputs[0] > 0:
                    rocket.apply_thrust()
                if outputs[1] > 0:
                    rocket.rotate_left()
                if outputs[2] > 0:
                    rocket.rotate_right()

                rocket.update()
                self.fitness_scores[i] += float(self.calculate_fitness(rocket))

        return all_done

    
    def calculate_fitness(self, rocket: Rocket) -> float:
        landing_pad_x = LANDING_X
        distance_to_pad = abs(rocket.x - landing_pad_x)

        # Base fitness components
        fitness = 0

        # Reward for being close to the landing pad
        fitness -= distance_to_pad * 1000

        # Penalize vertical velocity more aggressively
        fitness -= abs(rocket.vy) * 200000000

        # Penalize horizontal velocity
        fitness -= abs(rocket.vx) * 1000

        # Strong penalty for being off-angle
        fitness -= abs(rocket.angulo) * 2000

        # Reward remaining fuel (encourages efficient landing)
        fitness += (MAX_COMBUSTIVEL -  rocket.combustivel) * 10

        # Major success bonus
        if rocket.success:
            fitness += 10000

        # Severe penalties for failure modes
        if rocket.colidiu and not rocket.success:
            fitness -= 5000

        if rocket.hit_wall:
            fitness -= 50000

        return float(fitness)

    def sexual_evolve(self, top_performer_bias=2.0):
        # Reset rockets for the new generation
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.run = 0

        # Calculate the best fitness of the current generation
        current_max_fitness = max(self.fitness_scores)
        self.current_best_fitness = current_max_fitness
        
        if current_max_fitness > self.best_fitness:
            self.best_fitness = current_max_fitness
            self.best_network = self.population[self.fitness_scores.index(current_max_fitness)]

        # Sort networks by their fitness
        sorted_pairs = sorted(zip(self.fitness_scores, self.population),
                              key=lambda pair: pair[0], reverse=True)
        
        # Elite selection: Preserve the top performers
        elite_size = max(3, POPULATION_SIZE // 8)  # At least 2 elites
        elites = [network for _, network in sorted_pairs[:elite_size]]

        # Compute exponential fitness weights to strongly bias towards top performers
        remaining_fitness_scores = [f for f, _ in sorted_pairs[elite_size:]]
        if not remaining_fitness_scores:
            return  # Prevent division by zero or empty population

        # Exponential fitness scaling
        min_fitness = min(remaining_fitness_scores)
        normalized_fitness = [(f - min_fitness + 1) ** top_performer_bias 
                              for f in remaining_fitness_scores]
        total_fitness = sum(normalized_fitness)
        selection_probs = [f / total_fitness for f in normalized_fitness]

        def select_parent():
            # Include top performers in parent selection
            all_candidates = sorted_pairs[:elite_size // 2] + sorted_pairs[elite_size:]
            candidates = [network for _, network in all_candidates]
            weights = [1 / (i + 1) for i in range(len(all_candidates))]
            return random.choices(candidates, weights=weights, k=1)[0]

        # Create new population
        new_population = elites.copy()  # Start with elites

        # Ensure top performers reproduce multiple times
        top_performers = sorted_pairs[:elite_size // 2]
        for _, network in top_performers:
            for _ in range(2):  # Each top performer creates 2 offspring
                parent1 = network
                parent2 = select_parent()
                
                child = self.advanced_crossover(parent1, parent2)
                mutation_rate = self.compute_mutation_rate()
                child.mutate(rate=mutation_rate)
                
                new_population.append(child)

        # Fill remaining population
        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent()
            parent2 = select_parent()
            
            child = self.advanced_crossover(parent1, parent2)
            mutation_rate = self.compute_mutation_rate()
            child.mutate(rate=mutation_rate)
            
            new_population.append(child)

        # Replace population
        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1
        self.fitness_scores = [0.0] * POPULATION_SIZE

    def advanced_crossover(self, parent1, parent2):
        child = NeuralNetwork()
        for i in range(len(parent1.weights)):
            # More nuanced crossover strategy
            crossover_mask = np.random.random(parent1.weights[i].shape) < 0.5
            child.weights[i] = np.where(
                crossover_mask, 
                parent1.weights[i], 
                parent2.weights[i]
            )
        return child

    def compute_mutation_rate(self):
        # Adaptive mutation rate
        # Higher mutation early, lower as fitness improves
        base_rate = 0.1
        diversity_factor = len(set(tuple(map(tuple, net.weights[0])) for net in self.population)) / POPULATION_SIZE
        return base_rate * (1 - self.generation / 100) * diversity_factor

    def evolve(self):
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.run = 0
        print("evolving")

        max_fitness = max(self.fitness_scores)
        print(f"max fitness: {max_fitness}")
        self.current_best_fitness = max_fitness
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.best_network = self.population[self.fitness_scores.index(max_fitness)]

        sorted_pairs = sorted(zip(self.fitness_scores, self.population),
                              key=lambda pair: pair[0], reverse=True)
        sorted_networks = [x for _, x in sorted_pairs]

        elite_size = POPULATION_SIZE // 5
        new_population = sorted_networks[:elite_size]

        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(sorted_networks[:POPULATION_SIZE // 2])
            child = NeuralNetwork()
            child.weights = [w.copy() for w in parent.weights]
            child.mutate()
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        self.fitness_scores = [0.0] * POPULATION_SIZE  # Reset fitness scores

    def reset_run(self):
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]




def main():
    screen = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Rocket Landing AI")
    clock = pygame.time.Clock()

    evolution = Evolution()
    font = pygame.font.Font(None, 36)
    steps = 500
    runs = 10
    render = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("quitting")
                if evolution.best_network:
                    now = datetime.now()
                    time = now.strftime("%H:%M:%S")
                    filename = f"best_model_{time}.pkl"
                    with open(filename, "wb") as f:
                        pickle.dump(evolution.best_network.weights, f)
                    print(f"Best model saved to {filename}")

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    evolution.sexual_evolve()
                elif event.key == pygame.K_k:
                    if evolution.best_network:
                        now = datetime.now()
                        time = now.strftime("%H:%M:%S")
                        filename = f"best_model_{time}.pkl"
                        with open(filename, "wb") as f:
                            pickle.dump(evolution.best_network.weights, f)
                        print(f"Best model saved to {filename}")
                elif event.key == pygame.K_p:
                    steps = 1
                elif event.key == pygame.K_o:
                    steps = 500
                elif event.key == pygame.K_m:
                    render = False
                elif event.key == pygame.K_n:
                    render  = True


        for _ in range(steps):

            all_done = evolution.evaluate_step()

            if all_done:
                evolution.run += 1
                evolution.reset_run()

            if evolution.run == runs:
                evolution.sexual_evolve()

                global RAND_X_RANGE, RAND_Y_RANGE, RAND_ANGLE_RANGE, SPEED_LIMIT, RAND_VY_RANGE, RAND_VX_RANGE

                if evolution.generation % 10 == 0:
                    RAND_X_RANGE = min(300, RAND_X_RANGE + 1)
                    if evolution.generation > 150:
                        RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0.5)
                        RAND_VY_RANGE = min(10, RAND_VY_RANGE + 0.5)
                        RAND_VX_RANGE = min(10, RAND_VX_RANGE + 0.5)
                        runs = 20
                    SPEED_LIMIT = max(20, SPEED_LIMIT - 0.2)
                    # RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.01)

                    if evolution.generation > 500:
                        runs = 50

                
                if evolution.generation % 5 == 0:
                    # RAND_X_RANGE = min(200, RAND_X_RANGE + 0.5)
                    RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0)
                    # RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.02)

                if evolution.generation % 10 == 0:
                    # RAND_X_RANGE = min(200, RAND_X_RANGE + 1)
                    RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0)
                    # RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.05)



        
        screen.fill(PRETO)

        if render:
            pygame.draw.rect(screen, VERDE, (LANDING_X - 50, ALTURA - 10, 100, 10))
            for rocket in evolution.rockets:
                rocket.draw(screen)

        text = font.render(
            f"Gen {evolution.generation} All Time Best: {evolution.best_fitness:.2f} Current Best Fitness: {evolution.current_best_fitness:.2f}",
            True, BRANCO
        )
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        sys.exit()
