import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

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
POPULATION_SIZE = 20  # Reduced for better visualization
MUTATION_RATE = 0.1
MUTATION_RANGE = 0.05

class NeuralNetwork:
    def __init__(self):
        # Simple network: 6 inputs, 8 hidden neurons, 3 outputs
        self.weights1 = np.random.randn(6, 8)
        self.weights2 = np.random.randn(8, 3)
        
    def forward(self, inputs):
        # Simple feedforward without activation function for simplicity
        hidden = np.dot(inputs, self.weights1)
        output = np.dot(hidden, self.weights2)
        return output
    
    def mutate(self):
        # Random mutation of weights
        mask1 = np.random.random(self.weights1.shape) < MUTATION_RATE
        mask2 = np.random.random(self.weights2.shape) < MUTATION_RATE
        self.weights1 += mask1 * np.random.randn(*self.weights1.shape) * MUTATION_RANGE
        self.weights2 += mask2 * np.random.randn(*self.weights2.shape) * MUTATION_RANGE

class Rocket:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = LARGURA / 2
        self.y = ALTURA / 4
        self.angulo = random.uniform(-math.pi/4, math.pi/4)
        self.vx = random.uniform(-2, 2)
        self.vy = VELOCIDADE_INICIAL
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
        if not self.colidiu:
            self.vy += GRAVIDADE
            self.x += self.vx
            self.y += self.vy
            
            # Boundary checking
            if (self.x < 0 or self.x > LARGURA or self.y > ALTURA or 
                (self.y > ALTURA - 10 and (self.x < LARGURA/2 - 50 or self.x > LARGURA/2 + 50))):
                self.colidiu = True
                
            # Landing success check
            if (self.y > ALTURA - 20 and 
                LARGURA/2 - 50 < self.x < LARGURA/2 + 50 and 
                abs(self.vy) < 2 and 
                abs(self.angulo) < 0.2):
                self.success = True
                self.colidiu = True
        
    def draw(self, screen):
        if not self.colidiu:
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
        self.fitness_scores = [-math.inf] * POPULATION_SIZE
        
    def evaluate_step(self):
        # Single step evaluation for all rockets
        all_done = True
        for i, (rocket, network) in enumerate(zip(self.rockets, self.population)):
            if not rocket.colidiu:
                all_done = False
                # Create input vector
                landing_pad_x = LARGURA / 2
                inputs = np.array([
                    rocket.x / LARGURA,
                    rocket.y / ALTURA,
                    rocket.vx / 10,
                    rocket.vy / 10,
                    rocket.angulo / (math.pi/2),
                    (rocket.x - landing_pad_x) / LARGURA
                ])
                
                # Get network outputs
                outputs = network.forward(inputs)
                
                # Apply actions
                rocket.thrusting = False
                if outputs[0] > 0:
                    rocket.apply_thrust()
                if outputs[1] > 0:
                    rocket.rotate_left()
                if outputs[2] > 0:
                    rocket.rotate_right()
                    
                rocket.update()
                
                # Update fitness
                self.fitness_scores[i] = float(self.calculate_fitness(rocket))
                
        return all_done
    
    def calculate_fitness(self, rocket: Rocket) -> float:
        landing_pad_x = LARGURA / 2
        distance_to_pad = abs(rocket.x - landing_pad_x)
        
        fitness = -distance_to_pad
        fitness -= abs(rocket.vy) * 100
        fitness -= abs(rocket.vx) * 100
        fitness -= abs(rocket.angulo) * 50
        fitness += rocket.combustivel
        
        if rocket.success:
            fitness += 10000
            
        return float(fitness)
    
    def evolve(self):
        # Reset for new generation
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        
        # Update best network if we found a better one
        max_fitness = max(self.fitness_scores)
        print(f"max fitness: {max_fitness}")
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.best_network = self.population[self.fitness_scores.index(max_fitness)]
            
        # Selection and reproduction
        sorted_pairs = sorted(zip(self.fitness_scores, self.population), 
                            key=lambda pair: pair[0], reverse=True)
        sorted_networks = [x for _, x in sorted_pairs]
        
        # Keep best 20% unchanged
        elite_size = POPULATION_SIZE // 5
        new_population = sorted_networks[:elite_size]
        
        # Create rest through mutation
        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(sorted_networks[:POPULATION_SIZE//2])
            child = NeuralNetwork()
            child.weights1 = parent.weights1.copy()
            child.weights2 = parent.weights2.copy()
            child.mutate()
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1

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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill(PRETO)
        
        # Draw landing pad
        pygame.draw.rect(screen, VERDE, (LARGURA/2 - 50, ALTURA - 10, 100, 10))
        
        # Evaluate one step of all rockets
        all_done = evolution.evaluate_step()
        
        # Draw all rockets
        for rocket in evolution.rockets:
            rocket.draw(screen)
        
        # If all rockets are done, start new generation
        if all_done:
            evolution.evolve()
        
        # Draw stats
        gen_text = font.render(f"Generation: {evolution.generation}", True, BRANCO)
        fit_text = font.render(f"Best Fitness: {evolution.best_fitness:.0f}", True, BRANCO)
        screen.blit(gen_text, (10, 10))
        screen.blit(fit_text, (10, 50))
        
        # Check if any rocket succeeded in current generation
        if any(rocket.success for rocket in evolution.rockets):
            success_text = font.render("SUCCESS!", True, VERDE)
            screen.blit(success_text, (LARGURA/2 - 50, 10))
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
