import pygame
import random
import math

# Pygame initialisieren und Fenster erstellen
pygame.init()
WIDTH, HEIGHT = 150, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Massenpanik im Stall")
clock = pygame.time.Clock()
FPS = 60

# Parameter der Simulation
NUM_ANIMALS = 1000             # Gesamtanzahl der Tiere
NUM_CLUSTERS = 500             # Anzahl der anfänglichen Gruppen
ANIMAL_RADIUS = 5             # Darstellungsradius
PANIC_CLICK_RADIUS = 100      # Radius, in dem ein Mausklick Panik auslöst
PANIC_TRANSMISSION_DIST = 5  # Abstand, in dem Panik von Tier zu Tier übertragen wird
MAX_SPEED = 3                 # Maximale Geschwindigkeit
MAX_FORCE = 0.5               # Maximale Kraft, die auf ein Tier wirkt
REPULSION_DIST = 20           # Abstand für Abstoßungseffekt
REPULSION_FACTOR = 0.1        # Stärke der Abstoßung
ATTRACT_FORCE = 0.01           # Anziehende Kraft (Gruppenbildung)
FLEE_FORCE = 1.0              # Kraft, mit der panische Tiere von der Panikquelle fliehen
RANDOM_FORCE = 0.01     # Zufällige Kraft, die immer wirkt

# Parameter für Gruppenbildung beim Start
CLUSTER_SPREAD = 30           # Streuung der Tiere um das Gruppenzentrum
MARGIN = 5                   # Abstand zum Fensterrand, um Gruppen nicht zu nah am Rand zu platzieren

# Globale Variable für die Panikquelle (wird bei Mausklick gesetzt)
panic_origin = None

class Animal:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        # Geringe Anfangsgeschwindigkeit
        self.vel = pygame.math.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.panicked = False
        self.in_move = False

    def update(self, animals):
        force = pygame.math.Vector2(0, 0)
        # Kombinierte Berechnung von Abstoßungs- und Anziehungskräften
        for other in animals:
            if other is self:
                continue
            diff = self.pos - other.pos
            d_sq = diff.length_squared()

            if self.panicked:
                # Nur nahe Tiere (innerhalb von 100 Pixeln) berücksichtigen
                if 0 < d_sq < 100**2:
                    d = math.sqrt(d_sq)
                    # Abstoßung: wenn zu nah
                    if d_sq < REPULSION_DIST**2:
                        force += diff / d * (REPULSION_DIST - d) * REPULSION_FACTOR
                    # Anziehung: zur Bildung von Gruppen
                    force += (other.pos - self.pos) / d * ATTRACT_FORCE#
                    #random Bewegung
                    rand_move = random.uniform(-0.05, 0.05)
                    force += self.pos * rand_move * RANDOM_FORCE
                    

        # Wenn panisch, zusätzlich Fluchtkraft berechnen
        global panic_origin
        if self.panicked and panic_origin is not None:
            flee_dir = self.pos - panic_origin
            d_sq = flee_dir.length_squared()
            if d_sq > 0:
                d = math.sqrt(d_sq)
                force += flee_dir / d * FLEE_FORCE

        # Begrenze die resultierende Kraft
        if force.length() > MAX_FORCE:
            force.scale_to_length(MAX_FORCE)

        # Geschwindigkeit aktualisieren und auf MAX_SPEED begrenzen
        self.vel += force
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # Position aktualisieren
        self.pos += self.vel

        # Abprallen an den Fenstergrenzen
        if self.pos.x < ANIMAL_RADIUS or self.pos.x > WIDTH - ANIMAL_RADIUS:
            self.vel.x *= -1
        if self.pos.y < ANIMAL_RADIUS or self.pos.y > HEIGHT - ANIMAL_RADIUS:
            self.vel.y *= -1

    def draw(self, surface):
        # Panische Tiere werden rot, ruhige grün dargestellt
        color = (255, 0, 0) if self.panicked else (0, 255, 0)
        pygame.draw.circle(surface, color, (int(self.pos.x), int(self.pos.y)), ANIMAL_RADIUS)

# Erzeuge Tiere in zufällig verteilten Gruppen
animals = []
animals_per_cluster = NUM_ANIMALS // NUM_CLUSTERS
rest = NUM_ANIMALS % NUM_CLUSTERS

for _ in range(NUM_CLUSTERS):
    cluster_center = pygame.math.Vector2(
        random.randint(MARGIN, WIDTH - MARGIN),
        random.randint(MARGIN, HEIGHT - MARGIN)
    )
    group_size = animals_per_cluster + (1 if rest > 0 else 0)
    if rest > 0:
        rest -= 1
    for _ in range(group_size):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, CLUSTER_SPREAD)
        pos = cluster_center + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * radius
        animals.append(Animal(pos))

# Hauptprogrammschleife
running = True
while running:
    dt = clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Bei Mausklick: Panik auslösen und globale Panikquelle setzen
        elif event.type == pygame.MOUSEBUTTONDOWN:
            panic_origin = pygame.math.Vector2(event.pos)
            for animal in animals:
                if animal.pos.distance_to(panic_origin) < PANIC_CLICK_RADIUS:
                    animal.panicked = True

    # Panikübertragung: Ein panisches Tier infiziert ruhige Tiere in unmittelbarer Nähe
    for animal in animals:
        if animal.panicked:
            for other in animals:
                if not other.panicked:
                    if (animal.pos - other.pos).length_squared() < PANIC_TRANSMISSION_DIST**2:
                        other.panicked = True

    # Update der Tiere
    for animal in animals:
        animal.update(animals)

    # Zeichne Hintergrund und Tiere
    screen.fill((30, 30, 30))
    for animal in animals:
        animal.draw(screen)
    pygame.display.flip()

pygame.quit()







# #random Bewegung
# rand_move = random.uniform(-0.05, 0.05)
# force += self.pos * rand_move * RANDOM_FORCE