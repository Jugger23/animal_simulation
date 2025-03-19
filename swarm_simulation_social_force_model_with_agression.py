import pygame
import random
import math

# --- Pygame initialisieren und Fenster erstellen ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation: Aggression und Panik")
clock = pygame.time.Clock()
FPS = 60

# --- Simulationsparameter ---
NUM_ANIMALS = 700       # Gesamtzahl der Tiere
NUM_CLUSTERS = 200      # Viele Cluster für heterogene Verteilung
ANIMAL_RADIUS = 5       # Radius der Darstellung

# --- Parameter für die Verteilung der Tiere ---
CLUSTER_SPREAD = 20     # Streuung um das jeweilige Clusterzentrum

# --- Kräfte-Parameter ---
MAX_SPEED = 3           # Maximale Geschwindigkeit
MAX_FORCE = 0.5         # Maximale Summe aller Kräfte
REPULSION_DIST = 20     # Bereich, in dem repulsive Kräfte wirken
REPULSION_FACTOR = 0.1  # Basisstärke der Repulsionskraft
ATTRACT_FORCE = 0.1     # Basisstärke der Anziehungskraft (wird nur bei Panik voll aktiv)
PANIC_RANDOM_FORCE = 0.01  # Immer wirkende, zufällige Kraft

# --- Aggressionsparameter ---
AGG_RADIUS = 10         # Radius, in dem Interaktionen gezählt werden
AGG_THRESHOLD = 5       # Mindestens so viele Tiere in der Nähe, damit Aggression steigt
AGG_INCREASE_RATE = 0.005   # Langsame Zunahme des Aggressionslevels
AGG_DECAY_RATE = 0.5       # Schneller Abbau des Aggressionslevels, wenn wenige Interaktionen

# --- Tierklasse als Agent ---
class Animal:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        # Anfangsgeschwindigkeit klein, damit die Tiere nicht sofort wild umherflitzen
        self.vel = pygame.math.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.aggression = 0.0  # Aggressionslevel startet bei 0 (0%)
        self.panicked = False  # Panikstatus; wenn True, werden volle Kräfte aktiviert

    def update(self, animals):
        force = pygame.math.Vector2(0, 0)
        interaction_count = 0
        # Wir definieren einen maximal interessanten Radius, um teure Berechnungen zu vermeiden.
        max_threshold = max(AGG_RADIUS, 100, REPULSION_DIST)
        max_threshold_sq = max_threshold**2

        for other in animals:
            if other is self:
                continue
            diff = self.pos - other.pos
            d_sq = diff.length_squared()
            if d_sq > max_threshold_sq:
                continue

            # Zähle Interaktionen im Aggressionsradius
            if d_sq < AGG_RADIUS**2:
                interaction_count += 1

            # Wenn das Tier panisch ist, wirken volle Kräfte:
            if self.panicked:
                # Repulsive Kraft (voll)
                if d_sq < REPULSION_DIST**2:
                    d = math.sqrt(d_sq) if d_sq > 0 else 0.001
                    repulsion = diff.normalize() * (REPULSION_DIST - d) * REPULSION_FACTOR
                    force += repulsion
                # Zusätzlich: Anziehende Kraft (voll), um Gruppenbildung bzw. koordinierte Flucht zu ermöglichen
                if d_sq < 100**2:
                    d = math.sqrt(d_sq) if d_sq > 0 else 0.001
                    attraction = (other.pos - self.pos).normalize() * ATTRACT_FORCE
                    force += attraction
            else:
                # Nicht-panische Tiere: Nur repulsive Kraft verstärkt durch ihr Aggressionslevel
                if d_sq < REPULSION_DIST**2:
                    d = math.sqrt(d_sq) if d_sq > 0 else 0.001
                    repulsion = diff.normalize() * (REPULSION_DIST - d) * REPULSION_FACTOR
                    force += repulsion * self.aggression
                # Die anziehende Kraft bleibt aus, solange keine Panik vorliegt.

        # Aggressionslevel anpassen:
        # Bei vielen nahen Interaktionen steigt das Level langsam, ansonsten sinkt es schnell.
        if interaction_count >= AGG_THRESHOLD:
            self.aggression = min(1.0, self.aggression + AGG_INCREASE_RATE)
        else:
            self.aggression = max(0.0, self.aggression - AGG_DECAY_RATE)

        # Füge immer die randomisierte Kraft hinzu, damit Bewegung stattfindet
        random_force = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)) * PANIC_RANDOM_FORCE
        force += random_force

        # Begrenze die resultierende Kraft
        if force.length() > MAX_FORCE:
            force.scale_to_length(MAX_FORCE)

        # Geschwindigkeit aktualisieren und auf MAX_SPEED begrenzen
        self.vel += force
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # Position aktualisieren
        self.pos += self.vel

        # Abprallen an den Rändern des Fensters
        if self.pos.x < ANIMAL_RADIUS or self.pos.x > WIDTH - ANIMAL_RADIUS:
            self.vel.x *= -1
        if self.pos.y < ANIMAL_RADIUS or self.pos.y > HEIGHT - ANIMAL_RADIUS:
            self.vel.y *= -1

    def draw(self, surface):
        # Darstellung: Nicht-panisch: Farbe mischt grün (ruhig) und rot (bei erhöhter Aggression);
        # panische Tiere werden ganz rot.
        if self.panicked:
            color = (255, 0, 0)
        else:
            red = int(255 * self.aggression)
            green = 255 - red
            color = (red, green, 0)
        pygame.draw.circle(surface, color, (int(self.pos.x), int(self.pos.y)), ANIMAL_RADIUS)

# --- Tiere zu Beginn in vielen kleinen Clustern zufällig verteilen ---
cluster_centers = [pygame.math.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
                   for _ in range(NUM_CLUSTERS)]
animals = []
for i in range(NUM_ANIMALS):
    center = random.choice(cluster_centers)
    # Gaussisch verteilter Offset, damit nicht alle Tiere exakt im Zentrum eines Clusters landen
    offset = pygame.math.Vector2(random.gauss(0, CLUSTER_SPREAD), random.gauss(0, CLUSTER_SPREAD))
    pos = center + offset
    # Sicherstellen, dass das Tier im Fenster bleibt
    pos.x = max(ANIMAL_RADIUS, min(WIDTH - ANIMAL_RADIUS, pos.x))
    pos.y = max(ANIMAL_RADIUS, min(HEIGHT - ANIMAL_RADIUS, pos.y))
    animals.append(Animal(pos))

# --- Hauptprogrammschleife ---
running = True
while running:
    dt = clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Bei Mausklick wird Panik ausgelöst: Tiere im definierten Radius erhalten den Panikstatus,
        # wodurch sie volle attraktive und repulsive Kräfte nutzen.
        elif event.type == pygame.MOUSEBUTTONDOWN:
            panic_origin = pygame.math.Vector2(event.pos)
            for animal in animals:
                if animal.pos.distance_to(panic_origin) < 100:
                    animal.panicked = True

    # Update aller Tiere
    for animal in animals:
        animal.update(animals)

    # Zeichne Hintergrund und Tiere
    screen.fill((30, 30, 30))
    for animal in animals:
        animal.draw(screen)
    pygame.display.flip()

pygame.quit()
