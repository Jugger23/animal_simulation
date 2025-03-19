#ToDO's:
# 1. Implementieren Sie die Simulation von Tieren, die in einem Stall bleiben sollen.
# 2. Die Tiere sollen sich im Stall gleichmäßig verteilen und sich nicht überlappen.
# 3. Die Tiere sollen sich nicht durch Wände oder Ecken bewegen.
# 4. Die Tiere sollen sich nicht durch eine Stalltür bewegen.
# 5. Die Tiere sollen sich nicht durch die Stallwände bewegen.
# 6. Maximal 500 Tiere sollen sich im Stall befinden (3 Tiere pro m²).
# 7. (offen) Die Tiere sollen nicht auf der gleichen Position sein dürfen, wenn man die aktualisierte Position der Tiere nachträglich durch die Wände korrigiert.
# 8. (offen) Die Tiere werden an der Wand abprallen, wenn sie auf die Wand treffen.


import pygame
import numpy as np

# Fenster- und Raumparameter
WIDTH, HEIGHT = 150, 1000
NUM_ANIMALS = 700

# Simulationsparameter
dt = 0.05         # Zeitschritt (s)
tau = 0.5         # Relaxationszeit (s)
radius = 5        # Tier-Radius (Pixel)

# Parameter für lokale Interaktionskräfte
repulsion_range = 20    # Bereich, in dem repulsive Kräfte wirken
attraction_range = 100  # Bereich, in dem attractive Kräfte wirken

# Repulsive Kraft: vor der Panik abgeschaltet, nach der Panik stark
K_rep_prepanic = 0.0           # Konstante für repulsive Kraft
K_rep_panic = 100.0           # Konstante für repulsive Kraft

# Attractive Kraft: vor der Panik stark, nach der Panik abgeschaltet
K_attr_prepanic = 0.0
K_attr_panic = 100.0

# Kollisionskraft: vor der Panik abgeschaltet, nach der Panik stark
k_coll_prepanic = 0.0          # Federkonstante für Kollisionskraft
k_coll_panic = 100.0          # Federkonstante für Kollisionskraft

# PANIK-Parameter
PANIC_CLICK_RADIUS = 100.0  # Radius um den Klickpunkt, in dem Panik ausgelöst wird
PANIC_SPREAD_RADIUS = 20.0  # Radius, in dem Panik von panischen Tieren auf Nachbarn übergeht

# Geschwindigkeitsbegrenzungen
MAX_SPEED_PREPANIC = 3.0  # z. B. Huhngeschwindigkeit
MAX_SPEED_PANIC = 7.0     # Schneller bei Panik
v_desired = 20.0         # Basis-Wunschgeschwindigkeit bei Panik

# Reibung: Dämpfungskraft proportional zur Geschwindigkeit
friction_coef = 0.5

# Parameter zur Begrenzung der Kraft an Ecken
MAX_FORCE = 50.0

EPS = 1e-6

# # Initialisierung: Tiere werden gleichverteilt im Raum platziert
# pos = np.random.uniform([radius, radius], [WIDTH - radius, HEIGHT - radius], (NUM_ANIMALS, 2))
# vel = np.zeros((NUM_ANIMALS, 2))
# acc = np.zeros((NUM_ANIMALS, 2))
# panicked = np.zeros(NUM_ANIMALS, dtype=bool)
# panic_origin = np.zeros((NUM_ANIMALS, 2))  # Wird bei Panik gesetzt

# ----- Initiale Verteilung in Clustern (Gruppen) -----
NUM_CLUSTERS = 100
animals_per_cluster = NUM_ANIMALS // NUM_CLUSTERS
pos = np.zeros((NUM_ANIMALS, 2))
vel = np.zeros((NUM_ANIMALS, 2))
acc = np.zeros((NUM_ANIMALS, 2))
panicked = np.zeros(NUM_ANIMALS, dtype=bool)
panic_origin = np.zeros((NUM_ANIMALS, 2))  # Wird bei Panik gesetzt
# Definiere einen Sicherheitsabstand von den Wänden (z.B. 20 Pixel)
margin = 20
for cl in range(NUM_CLUSTERS):
    # Wähle eine zufällige Cluster-Mitte im inneren Bereich
    center = np.random.uniform([margin, margin], [WIDTH - margin, HEIGHT - margin])
    idx_start = cl * animals_per_cluster
    idx_end = (cl + 1) * animals_per_cluster
    # Verteile die Tiere innerhalb eines kleinen Bereichs um die Cluster-Mitte (Normalverteilung)
    pos[idx_start:idx_end] = np.random.normal(loc=center, scale=20, size=(animals_per_cluster, 2))
# Falls es Reste gibt, verteile sie gleichmäßig
if NUM_ANIMALS % NUM_CLUSTERS != 0:
    pos[-(NUM_ANIMALS % NUM_CLUSTERS):] = np.random.uniform([margin, margin], [WIDTH - margin, HEIGHT - margin], (NUM_ANIMALS % NUM_CLUSTERS, 2))



pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation: Tiere bleiben im Stall")
clock = pygame.time.Clock()

running = True
while running:
    # Ereignisbehandlung
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_pos = np.array(pygame.mouse.get_pos(), dtype=float)
            distances = np.linalg.norm(pos - click_pos, axis=1)
            to_panic = distances < PANIC_CLICK_RADIUS
            panicked[to_panic] = True
            panic_origin[to_panic] = click_pos

    # Lokale Panikübertragung
    non_panicked_idx = np.where(~panicked)[0]
    panicked_idx = np.where(panicked)[0]
    if panicked_idx.size > 0 and non_panicked_idx.size > 0:
        d_mat = np.linalg.norm(pos[non_panicked_idx][:, None, :] - pos[panicked_idx][None, :, :], axis=2)
        propagate = (d_mat < PANIC_SPREAD_RADIUS).any(axis=1)
        indices_to_update = non_panicked_idx[propagate]
        if indices_to_update.size > 0:
            panic_origin[indices_to_update] = panic_origin[panicked_idx[0]]
            panicked[indices_to_update] = True

    # Vektorisierte Berechnung der paarweisen Differenzen
    diff = pos[:, None, :] - pos[None, :, :]       # (N, N, 2)
    d = np.linalg.norm(diff, axis=2)                # (N, N)
    np.fill_diagonal(d, np.inf)                     # Selbstinteraktion ignorieren
    d_safe = np.where(d < EPS, EPS, d)
    e = diff / d_safe[:, :, None]

    rep_mask = d < repulsion_range
    K_rep_effective = np.where(panicked[:, None], K_rep_panic, K_rep_prepanic)
    F_rep = np.where(rep_mask[:, :, None],
                   K_rep_effective * (repulsion_range - d_safe)[:, :, None] / (d_safe**2)[:, :, None] * (-e),
                   0)

    attr_mask = (d < attraction_range) & (d >= repulsion_range)
    K_attr_array = np.where(panicked, K_attr_panic, K_attr_prepanic)
    F_attr = np.where(attr_mask[:, :, None],
                      (K_attr_array[:, None] / (d_safe**2)[:, :, None]) * e,
                      0)

    coll_mask = d < (2 * radius)
    K_coll_effective = np.where(panicked[:, None], k_coll_panic, k_coll_prepanic)
    F_coll = np.where(coll_mask[:, :, None],
                  K_coll_effective * ((2 * radius - d_safe))[:, :, None] * (-e),
                  0)

    F_int = np.sum(F_rep + F_attr + F_coll, axis=1)
    F_rand = np.random.uniform(-1, 1, (NUM_ANIMALS, 2))
    F_imp = np.zeros((NUM_ANIMALS, 2))
    panicked_indices = np.where(panicked)[0]
    if panicked_indices.size > 0:
        diff_panic = pos[panicked_indices] - panic_origin[panicked_indices]
        dist_panic = np.linalg.norm(diff_panic, axis=1, keepdims=True)
        d_panic_safe = np.where(dist_panic < EPS, EPS, dist_panic)
        e_panic = diff_panic / d_panic_safe
        desired_vel = v_desired * e_panic
        F_imp[panicked_indices] = (desired_vel - vel[panicked_indices]) / tau

    F_fric = -friction_coef * vel

    F_total = F_int + F_rand + F_imp + F_fric
    F_total = np.minimum(F_total, MAX_FORCE)

    # --- Wand- und Eckenanpassung ---
    epsilon_wall = 1e-3
    for i in range(NUM_ANIMALS):
        in_left = pos[i, 0] <= radius + epsilon_wall
        in_right = pos[i, 0] >= WIDTH - radius - epsilon_wall
        in_top = pos[i, 1] <= radius + epsilon_wall
        in_bottom = pos[i, 1] >= HEIGHT - radius - epsilon_wall
        if (in_left or in_right) and (in_top or in_bottom):
            corner_x = radius if in_left else WIDTH - radius
            corner_y = radius if in_top else HEIGHT - radius
            corner = np.array([corner_x, corner_y])
            diff_corner = pos[i] - corner
            norm_diff = np.linalg.norm(diff_corner)
            if norm_diff > EPS:
                n = diff_corner / norm_diff
                f_component = np.dot(F_total[i], n)
                if f_component > MAX_FORCE:
                    excess = f_component - MAX_FORCE
                    F_total[i] -= excess * n
                    # vel[i] = 0.0  # Tier verharrt in der Ecke

    # Zustandsaktualisierung
    acc = F_total
    vel += acc * dt
    pos += vel * dt

    # --- Sicherstellen, dass Tiere im Stall bleiben ---
    pos[:, 0] = np.clip(pos[:, 0], radius, WIDTH - radius)      # debug
    pos[:, 1] = np.clip(pos[:, 1], radius, HEIGHT - radius)     # debug

    speeds = np.linalg.norm(vel, axis=1)
    max_speed = np.where(panicked, MAX_SPEED_PANIC, MAX_SPEED_PREPANIC)
    factor = np.minimum(1, max_speed / (speeds + EPS))
    vel = (vel.T * factor).T

    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (0, 0), (WIDTH, 0), 5)
    pygame.draw.line(screen, (0, 0, 0), (0, HEIGHT), (WIDTH, HEIGHT), 5)
    pygame.draw.line(screen, (0, 0, 0), (0, 0), (0, HEIGHT), 5)
    pygame.draw.line(screen, (0, 0, 0), (WIDTH, 0), (WIDTH, HEIGHT), 5)
    for i in range(NUM_ANIMALS):
        color = (255, 0, 0) if panicked[i] else (0, 255, 0)
        pygame.draw.circle(screen, color, (int(pos[i, 0]), int(pos[i, 1])), radius)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
