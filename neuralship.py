import math
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg

pygame.init()
pygame.time
# Configurações da janela de simulação
LARGURA = 800  # Largura da tela
ALTURA = 600   # Altura da tela
# Definição de cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)

# Parâmetros físicos da simulação
GRAVIDADE = 0.5  # Aceleração da gravidade
IMPULSO = 2.0    # Força de impulso do foguete
RAPIDEZ_ROTACAO = 0.1  # Velocidade de rotação
MAX_COMBUSTIVEL = 100  # Quantidade máxima de combustível
VELOCIDADE_INICIAL = 2  # Velocidade inicial

# Configurações de população e algoritmo genético
POPULATION_SIZE = 50   # Tamanho da população
MUTATION_RATE = 0.5    # Taxa de mutação
MUTATION_RANGE = 1     # Alcance da mutação

# Configurações da rede neural
INPUT_SIZE = 7  
HIDDEN_LAYERS = [32, 32]  
OUTPUT_SIZE = 3
RAND_X_RANGE = 0
RAND_Y_RANGE = 0
RAND_VX_RANGE = 0
RAND_VY_RANGE = 0
RAND_ANGLE_RANGE = 0.5
SPEED_LIMIT = 30
average_fitness = []
global runs
runs = 10

# STEPS_PER_FRAME = 50


class NeuralNetwork:
    def __init__(self):
        # Inicialização da rede neural com pesos aleatórios
        self.weights = []
        layer_sizes = [INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_SIZE]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, inputs):
        # Propagação para frente na rede neural usando tangente hiperbólica
        activation = inputs
        for weight in self.weights:
            activation = np.dot(activation, weight)
            activation = np.tanh(activation) 
        return activation

    def mutate(self, rate=MUTATION_RATE):
        # Mutação dos pesos da rede neural
        for i in range(len(self.weights)):
            mask = np.random.random(self.weights[i].shape) < rate
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * MUTATION_RANGE


class Rocket:
    def __init__(self):
        # Inicialização do foguete
        self.reset()

  

    def reset(self):
        # Reinicialização das condições do foguete
        # Ângulo inicial aleatório próximo de zero
        self.angulo = random.uniform(-RAND_ANGLE_RANGE, RAND_ANGLE_RANGE)

        # Velocidade base com variação aleatória
        base_speed = 10 + random.uniform(-RAND_VY_RANGE, RAND_VY_RANGE)
         
        # Componentes x e y da velocidade baseadas no ângulo
        self.vx = base_speed * math.sin(self.angulo) 
        self.vy = abs(base_speed * math.cos(self.angulo)) 
        
        # Cálculo da trajetória inicial
        trajectory_time = 1.0  
        
        # Definição de um alvo próximo ao centro da tela
        target_x = LARGURA / 2 + random.uniform(-RAND_X_RANGE, RAND_X_RANGE)
        target_y = ALTURA / 4 + random.uniform(-RAND_Y_RANGE, RAND_Y_RANGE)
        
        # Posição inicial calculada retroativamente
        self.x = target_x - (self.vx * trajectory_time)
        self.y = target_y - (self.vy * trajectory_time - 0.5 * GRAVIDADE * trajectory_time * trajectory_time)
        
        # Reinicialização de estados
        self.combustivel = MAX_COMBUSTIVEL
        self.colidiu = False
        self.fitness = 0
        self.success = False
        self.thrusting = False
        self.hit_wall = False# How far back in time to start

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
    # Verifica se o foguete não colidiu e não completou o pouso
        if not self.colidiu and not self.success:
            # Aplica gravidade na velocidade vertical
            self.vy += GRAVIDADE
            
            # Atualiza a posição do foguete baseado nas velocidades
            self.x += self.vx
            self.y += self.vy

            # Verifica condições de colisão com os limites da tela
            if (self.x < 0 or self.x > LARGURA or self.y > ALTURA or
                (self.y > ALTURA - 10 and (self.x < LARGURA / 2 - 50 or self.x > LARGURA / 2 + 50))):
                
                # Se bater nos limites laterais ou superiores, marca como colisão
                if(self.x < 0 or self.x > LARGURA or self.y < 0):
                    self.hit_wall = True
                    self.vy = 10000  # Força uma penalização extrema para colisões com paredes

                self.colidiu = True

            # Verifica condições de pouso bem-sucedido
            if (self.y > ALTURA - 20 and
                LARGURA / 2 - 50 < self.x < LARGURA / 2 + 50 and
                abs(self.vy) < SPEED_LIMIT and
                abs(self.angulo) < 0.2):
                # Marca como pouso bem-sucedido
                self.success = True
                self.colidiu = True

    def draw(self, screen):
        if not self.colidiu or self.success:
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
        # Inicializa população de redes neurais e foguetes
        self.population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
        # Variáveis de acompanhamento da evolução
        self.generation = 1
        self.best_fitness = float('-inf')
        self.best_network = None
        # Pontuações de aptidão para cada indivíduo
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.fitness_scores = [0.0] * POPULATION_SIZE
        self.current_best_fitness = float('-inf')
        self.run = 0

    def evaluate_step(self):
        # Verifica se todos os foguetes terminaram sua trajetória
        all_done = True
        # Itera sobre cada foguete e sua respectiva rede neural
        for i, (rocket, network) in enumerate(zip(self.rockets, self.population)):
            # Se o foguete ainda está em movimento
            if not rocket.colidiu:
                all_done = False
                # Posição da plataforma de pouso
                landing_pad_x = LARGURA / 2

                # Prepara entradas para a rede neural
                # Normaliza informações para valores entre 0 e 1
               
                inputs = np.array([
                    rocket.x / LARGURA,                     # Posição x normalizada
                    rocket.y / ALTURA,                      # Posição y normalizada
                    rocket.vx / 10,                         # Velocidade x normalizada
                    rocket.vy / 10,                         # Velocidade y normalizada
                    rocket.angulo / (math.pi / 2),          # Ângulo normalizado
                    (rocket.x - landing_pad_x) / LARGURA,   # Distância da plataforma
                    rocket.combustivel / MAX_COMBUSTIVEL    # Combustível restante
                ])
                # Obtém saídas da rede neural
                outputs = network.forward(inputs)
                # Reinicia estado de propulsão
                rocket.thrusting = False

                # Decide ações baseadas nas saídas da rede
                if outputs[0] > 0:
                    rocket.apply_thrust() # Aplica impulso
                if outputs[1] > 0:
                    rocket.rotate_left() # Rotaciona para esquerda
                if outputs[2] > 0:
                    rocket.rotate_right() # Rotaciona para direita

                # Atualiza estado do foguete
                rocket.update()

                # Calcula e acumula pontuação de aptidão
                self.fitness_scores[i] += float(self.calculate_fitness(rocket))

        return all_done

    
    def calculate_fitness(self, rocket: Rocket) -> float:
        landing_pad_x = LARGURA / 2
        distance_to_pad = abs(rocket.x - landing_pad_x)

        fitness = 0

        fitness -= distance_to_pad * 1000

        fitness -= abs(rocket.vy) * 200000000

        fitness -= abs(rocket.vx) * 1000

        fitness -= abs(rocket.angulo) * 2000

        fitness += (MAX_COMBUSTIVEL -  rocket.combustivel) * 10

        if rocket.success:
            fitness += 10000

        if rocket.colidiu and not rocket.success:
            fitness -= 5000

        if rocket.hit_wall:
            fitness -= 50000

        return float(fitness)

    def sexual_evolve(self, top_performer_bias=2.0, cull_percentage=0.1):
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.run = 0
        current_max_fitness = max(self.fitness_scores)
        average_fitness.append(current_max_fitness / runs)
        self.current_best_fitness = current_max_fitness
        
        if current_max_fitness > self.best_fitness:
            self.best_fitness = current_max_fitness
            self.best_network = self.population[self.fitness_scores.index(current_max_fitness)]
        

        cull_count = int(POPULATION_SIZE * cull_percentage)
        

        sorted_pairs = sorted(zip(self.fitness_scores, self.population),
                              key=lambda pair: pair[0], reverse=True)
        

        self.population = [network for _, network in sorted_pairs[:-cull_count]]
        

        elite_size = max(3, len(self.population) // 8)  
        

        elites = sorted_pairs[:-cull_count][:elite_size]
        elites = [network for _, network in elites]
        

        remaining_fitness_scores = [f for f, _ in sorted_pairs[:-cull_count][elite_size:]]

        if not remaining_fitness_scores:
            return  
        
        min_fitness = min(remaining_fitness_scores)
        normalized_fitness = [(f - min_fitness + 1) ** top_performer_bias 
                              for f in remaining_fitness_scores]
        total_fitness = sum(normalized_fitness)
        selection_probs = [f / total_fitness for f in normalized_fitness]
        
        def select_parent():

            all_candidates = sorted_pairs[:-cull_count][:elite_size // 2] + sorted_pairs[:-cull_count][elite_size:]
            candidates = [network for _, network in all_candidates]
            weights = [1 / (i + 1) for i in range(len(all_candidates))]
            return random.choices(candidates, weights=weights, k=1)[0]
        

        new_population = elites.copy()  
        

        top_performers = sorted_pairs[:-cull_count][:elite_size // 2]
        for _, network in top_performers:
            for _ in range(2):  
                parent1 = network
                parent2 = select_parent()
                
                child = self.advanced_crossover(parent1, parent2)
                mutation_rate = self.compute_mutation_rate()
                child.mutate(rate=mutation_rate)
                
                new_population.append(child)
        

        while len(new_population) < POPULATION_SIZE:

            if len(new_population) > POPULATION_SIZE - cull_count:
                new_network = NeuralNetwork() 
                new_population.append(new_network)
            else:

                parent1 = select_parent()
                parent2 = select_parent()
                
                child = self.advanced_crossover(parent1, parent2)
                mutation_rate = self.compute_mutation_rate()
                child.mutate(rate=mutation_rate)
                
                new_population.append(child)
        

        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1
        self.fitness_scores = [0.0] * POPULATION_SIZE

    def advanced_crossover(self, parent1, parent2):
        child = NeuralNetwork()
        for i in range(len(parent1.weights)):
            crossover_mask = np.random.random(parent1.weights[i].shape) < 0.5
            child.weights[i] = np.where(
                crossover_mask, 
                parent1.weights[i], 
                parent2.weights[i]
            )
        return child

    def compute_mutation_rate(self):
        base_rate = 0.1
        diversity_factor = len(set(tuple(map(tuple, net.weights[0])) for net in self.population)) / POPULATION_SIZE
        return base_rate * (1 - self.generation / 300) * diversity_factor

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
        self.fitness_scores = [0.0] * POPULATION_SIZE  

    def reset_run(self):
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]

def plot_fitness_history(fitness_history):
    plt.figure(figsize=(3, 2), dpi=50, facecolor='black')
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    plt.plot(fitness_history, color='orange')  
    plt.title('Best Fitness', fontsize=8, color='white')
    plt.xlabel('Generation', fontsize=6, color='white')
    plt.ylabel('Fitness', fontsize=6, color='white')
    plt.tick_params(axis='both', which='major', labelsize=6, colors='white')
    
    # Add grid for better readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return plt.gcf()
    
def plot_to_surface(delicious_fig):
    canvas = FigureCanvasAgg(delicious_fig)
    canvas.draw()
    plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
    plt.close(delicious_fig)
    return plot_surface




def main():
    screen = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Rocket Landing AI")
    clock = pygame.time.Clock()

    evolution = Evolution()
    font = pygame.font.Font(None, 36)
    steps = 500
    render = True
    fitness_plot_surface = None
    global runs
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
                    print(f"Best model saved to {filename}")# Save fitness history plotplt.figure(figsize=(3, 2), dpi=50, facecolor='black')
                    plt.figure(figsize=(30, 20), dpi=100, facecolor='black')
                    plt.rcParams['axes.facecolor'] = 'black'
                    plt.rcParams['text.color'] = 'white'
                    plt.rcParams['axes.labelcolor'] = 'white'
                    plt.rcParams['xtick.color'] = 'white'
                    plt.rcParams['ytick.color'] = 'white'

                    plt.plot(average_fitness, color='orange')  
                    plt.title('Best Fitness', fontsize=20, color='white')
                    plt.xlabel('Generation', fontsize=20, color='white')
                    plt.ylabel('Fitness', fontsize=20, color='white')
                    plt.tick_params(axis='both', which='major', labelsize=20, colors='white')
                    
                    # Add grid for better readability
                    plt.grid(color='gray', linestyle='--', linewidth=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(f"fitness_history_{time}.png")
                    print(f"Fitness history saved to fitness_history_{time}.png")

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

                if evolution.generation > 500:
                    RAND_X_RANGE = 200
                    RAND_Y_RANGE = 100

                if evolution.generation % 10 == 0:
                    RAND_X_RANGE = min(300, RAND_X_RANGE + 1)
                    if evolution.generation > 150:
                        RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0.5)
                        RAND_VY_RANGE = min(20, RAND_VY_RANGE + 0.5)
                        RAND_VX_RANGE = min(20, RAND_VX_RANGE + 0.5)
                        runs = 20
                    SPEED_LIMIT = max(20, SPEED_LIMIT - 0.2)
                    RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.01)

                    if evolution.generation > 500:
                        runs = 50


                if evolution.generation % 5 == 0:
                    RAND_X_RANGE = min(200, RAND_X_RANGE + 0.5)
                    RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0)
                    RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.02)

                if evolution.generation % 10 == 0:
                    RAND_X_RANGE = min(200, RAND_X_RANGE + 1)
                    RAND_Y_RANGE = min(50, RAND_Y_RANGE + 0)
                    RAND_ANGLE_RANGE = min(math.pi, RAND_ANGLE_RANGE + 0.05)
#


        
        screen.fill(PRETO)

        if evolution.generation % 5 == 0:
            fitness_plot = plot_fitness_history(average_fitness)
            fitness_plot_surface = plot_to_surface(fitness_plot)

        if render:
            pygame.draw.rect(screen, VERDE, (LARGURA / 2 - 50, ALTURA - 10, 100, 10))
            for rocket in evolution.rockets:
                rocket.draw(screen)

            if fitness_plot_surface:
                screen.blit(fitness_plot_surface, (10, 50))

        text = font.render(
            f"Gen {evolution.generation}",
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
