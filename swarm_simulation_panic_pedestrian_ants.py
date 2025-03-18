import pygame
import numpy as np
import random
import math

# Fenster- und Raumparameter
WIDTH, HEIGHT = 800, 600
DOOR_WIDTH = 60  # Türhöhe (am rechten Rand)
EXIT_POS = np.array([WIDTH, HEIGHT/2])  # Position des Ausgangs (Türmitte)
NUM_ANIMALS = 50

# Simulationsparameter
dt = 0.05           # Zeitschritt in Sekunden
v_desired = 20.0     # Wunschgeschwindigkeit (Einheiten pro Sekunde)
tau = 0.5           # Relaxationszeit (s)
mass = 1.0          # Masse (Einheitlich angenommen)
radius = 5          # Radius jedes Tieres (in Pixeln)

# Parameter für Interaktionskräfte
repulsion_range = 20    # Schwelle für repulsive Wechselwirkung
attraction_range = 100  # Schwelle für attraktive Wechselwirkung
K_rep = 100.0           # Konstante für repulsive Kraft
K_attr = 50.0           # Konstante für attraktive Kraft
k_coll = 200.0          # Federkonstante für Kollisions-/Pushing-Kraft

class Animal:
    def __init__(self, pos):
        self.pos = pos.astype(float)
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.mass = mass
        self.radius = radius
    
    def update(self, animals):
        force = np.array([0.0, 0.0])
        
        # Impulsive Kraft: Tiere wollen zum Ausgang (siehe Formel 7)
        direction_to_exit = EXIT_POS - self.pos
        dist_to_exit = np.linalg.norm(direction_to_exit)
        if dist_to_exit > 0:
            e_exit = direction_to_exit / dist_to_exit
        else:
            e_exit = np.array([0.0, 0.0])
        desired_velocity = v_desired * e_exit
        F_imp = (desired_velocity - self.vel) / tau  # Da m=1 gilt: a = F_imp
        force += F_imp
        
        # Lokale interaktive Kräfte (repulsiv/attraktiv) und Kollisionskräfte
        for other in animals:
            if other is self:
                continue
            d_vec = other.pos - self.pos
            d = np.linalg.norm(d_vec)
            if d > 0:
                e = d_vec / d
            else:
                continue

            # Repulsive Kraft: bei zu kleinen Abständen
            if d < repulsion_range:
                # Kraft wirkt weg vom anderen Tier (inverses Quadratgesetz)
                F_rep = K_rep * (repulsion_range - d) / (d**2) * (-e)
                force += F_rep
            # Attraktive Kraft: wenn der Abstand zwischen repulsiver und attraktiver Zone liegt
            elif d < attraction_range:
                F_attr = K_attr / (d**2) * e
                force += F_attr
                
            # Kollisionskraft: Falls sich die Kreise überlappen, einfache Federkraft
            overlap = self.radius + other.radius - d
            if overlap > 0:
                F_coll = k_coll * overlap * (-e)
                force += F_coll
                
        # Zufallskomponente (Randomness)
        random_force = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        force += random_force
        
        # Gesamte Beschleunigung (F = m * a, mit m = 1)
        self.acc = force
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        
        # Wandkollisionen: Der Raum ist umschlossen – außer an der Tür (rechter Rand)
        # Linke Wand:
        if self.pos[0] < self.radius:
            self.pos[0] = self.radius
            self.vel[0] = -self.vel[0]
        # Obere Wand:
        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] = -self.vel[1]
        # Untere Wand:
        if self.pos[1] > HEIGHT - self.radius:
            self.pos[1] = HEIGHT - self.radius
            self.vel[1] = -self.vel[1]
        # Rechte Wand (außer Türbereich):
        if self.pos[0] > WIDTH - self.radius:
            if not ((self.pos[1] > (HEIGHT/2 - DOOR_WIDTH/2)) and (self.pos[1] < (HEIGHT/2 + DOOR_WIDTH/2))):
                self.pos[0] = WIDTH - self.radius
                self.vel[0] = -self.vel[0]
        
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.pos[0]), int(self.pos[1])), self.radius)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulation: Tierbewegung in Panik")
    clock = pygame.time.Clock()
    
    # Initialisierung: Tiere werden zufällig im linken Raumteil verteilt
    animals = []
    for i in range(NUM_ANIMALS):
        x = random.uniform(50, WIDTH/2)
        y = random.uniform(50, HEIGHT-50)
        animals.append(Animal(np.array([x, y])))
        
    running = True
    simulation_running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if simulation_running:
            # Update aller Tiere
            for animal in animals:
                animal.update(animals)
            # Tiere, die den Ausgang erreicht haben, entfernen
            animals = [a for a in animals if not (
                a.pos[0] >= WIDTH - a.radius and 
                (a.pos[1] > (HEIGHT/2 - DOOR_WIDTH/2)) and 
                (a.pos[1] < (HEIGHT/2 + DOOR_WIDTH/2))
            )]
            if len(animals) == 0:
                simulation_running = False
        
        # Zeichnen des Hintergrunds und der Wände
        screen.fill((255, 255, 255))
        # Linke, obere und untere Wand als Linien
        pygame.draw.line(screen, (0,0,0), (0,0), (0, HEIGHT), 5)
        pygame.draw.line(screen, (0,0,0), (0,0), (WIDTH, 0), 5)
        pygame.draw.line(screen, (0,0,0), (0, HEIGHT), (WIDTH, HEIGHT), 5)
        # Rechte Wand: Türbereich ausgenommen (Tür als Lücke in der rechten Wand)
        pygame.draw.line(screen, (0,0,0), (WIDTH, 0), (WIDTH, int(HEIGHT/2 - DOOR_WIDTH/2)), 5)
        pygame.draw.line(screen, (0,0,0), (WIDTH, int(HEIGHT/2 + DOOR_WIDTH/2)), (WIDTH, HEIGHT), 5)
        
        # Zeichne alle Tiere
        for animal in animals:
            animal.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
        
    pygame.quit()

if __name__ == '__main__':
    main()
