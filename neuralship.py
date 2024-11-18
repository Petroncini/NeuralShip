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
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
MUTATION_RANGE = 0.5

# Neural Network architecture
INPUT_SIZE = 7  # Adjusted for the new inputs
HIDDEN_LAYERS = [32, 32]  # Can be adjusted to increase complexity
OUTPUT_SIZE = 3
RAND_X_RANGE = 0
RAND_Y_RANGE = 0
RAND_ANGLE_RANGE = math.pi/6


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

    def mutate(self):
        # Apply random mutations to weights
        for i in range(len(self.weights)):
            mask = np.random.random(self.weights[i].shape) < MUTATION_RATE
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * MUTATION_RANGE


class Rocket:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = LARGURA / 2 +random.uniform(-RAND_X_RANGE, RAND_X_RANGE)
        self.y = ALTURA / 4 - 90 + random.uniform(-RAND_Y_RANGE, RAND_Y_RANGE)
        self.angulo = random.uniform(-RAND_ANGLE_RANGE, RAND_ANGLE_RANGE)
        self.vx = 0
        self.vy = 5
        self.combustivel = MAX_COMBUSTIVEL
        self.colidiu = False
        self.fitness = 0
        self.success = False
        self.thrusting = False  # Added for flame visualization

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
                (self.y > ALTURA - 10 and (self.x < LARGURA / 2 - 50 or self.x > LARGURA / 2 + 50))):
                self.colidiu = True

            # Landing success check
            if (self.y > ALTURA - 20 and
                LARGURA / 2 - 50 < self.x < LARGURA / 2 + 50 and
                abs(self.vy) < 30 and
                abs(self.angulo) < 0.2) and self.combustivel < 90:
                self.success = True
                self.colidiu = True
                print("success!")
                print(f"fuel: {self.combustivel}")

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
                landing_pad_x = LARGURA / 2
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
        landing_pad_x = LARGURA / 2
        distance_to_pad = abs(rocket.x - landing_pad_x)

        fitness = -distance_to_pad
        fitness -= abs(rocket.vy) * 100
        fitness -= abs(rocket.vx) * 100
        fitness -= abs(rocket.angulo) * 50
        fitness += rocket.combustivel

        if(rocket.colidiu):
            fitness -= 500

        if rocket.success:
            fitness += 1000

        return float(fitness)

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        child = NeuralNetwork()
        for i in range(len(parent1.weights)):
            # Perform crossover for each weight matrix
            mask = np.random.rand(*parent1.weights[i].shape) > 0.5  # 50% chance per weight
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return child


    def sexual_evolve(self):
        # Reset rockets for the new generation
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]

        # Calculate the best fitness of the current generation
        current_max_fitness = max(self.fitness_scores)
        self.current_best_fitness = current_max_fitness
        if current_max_fitness > self.best_fitness:
            self.best_fitness = current_max_fitness
            self.best_network = self.population[self.fitness_scores.index(current_max_fitness)]

        # Sort networks by their fitness
        sorted_pairs = sorted(zip(self.fitness_scores, self.population),
                              key=lambda pair: pair[0], reverse=True)
        sorted_networks = [x for _, x in sorted_pairs]

        # Elite selection: Preserve the top k networks
        elite_size = min(5, POPULATION_SIZE)  # Ensure we don't exceed the population size
        elites = sorted_networks[:elite_size]  # These are NeuralNetwork objects

        # Normalize fitness scores for the remaining individuals
        remaining_fitness_scores = [f for f, _ in sorted_pairs[elite_size:]]
        min_fitness = min(remaining_fitness_scores) if remaining_fitness_scores else 0
        fitness_range = max(1.0, current_max_fitness - min_fitness)
        normalized_fitness = [
            (f - min_fitness) / fitness_range
            for f, _ in sorted_pairs[elite_size:]
        ]

        # Selection pool for the rest of the population
        selection_probs = [f / sum(normalized_fitness) for f in normalized_fitness]

        def select_parent():
            return random.choices(sorted_networks[elite_size:], weights=selection_probs, k=1)[0]

        # Create the rest of the new population using sexual reproduction
        new_population = elites  # Start with elites
        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent()
            parent2 = select_parent()
            child = self.crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)

        # Replace the population with the new one
        self.population = new_population
        self.generation += 1
        self.fitness_scores = [0.0] * POPULATION_SIZE  # Reset fitness scores


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
                    evolution.evolve()
                elif event.key == pygame.K_k:
                    if evolution.best_network:
                        now = datetime.now()
                        time = now.strftime("%H:%M:%S")
                        filename = f"best_model_{time}.pkl"
                        with open(filename, "wb") as f:
                            pickle.dump(evolution.best_network.weights, f)
                        print(f"Best model saved to {filename}")

        screen.fill(PRETO)
        pygame.draw.rect(screen, VERDE, (LARGURA / 2 - 50, ALTURA - 10, 100, 10))

        all_done = evolution.evaluate_step()

        if all_done:
            evolution.run += 1
            evolution.reset_run()

        if evolution.run == 2:
            evolution.evolve()

            global RAND_X_RANGE, RAND_Y_RANGE, RAND_ANGLE_RANGE

            if evolution.generation % 5 == 0:
                RAND_X_RANGE = min(200, RAND_X_RANGE + 2)
                RAND_Y_RANGE = min(50, RAND_Y_RANGE + 2)
                RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.1)


        


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
