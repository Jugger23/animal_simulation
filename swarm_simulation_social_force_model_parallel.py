# TODO
# - Erl.: start condition: starting velocity is uniform which moves the animals all to the borders
# - Erl.: flee force: the flee force is not working properly. flee force is very fast the only active force.
# - Erl.: position of one aninmal is independent of the others: animals can be on the same position
# - Erl.: Panic status not working properly: Panic status is not being updated properly. It is not being updated when the panic source is removed.
# - Erl.: Random force: approach to include the random force without adding it in every iteration. Just add it once in the beginning and adjust it radnomly so the movement can switch to other directions and veloctities abruptly.
# - Erl.: Add random force at the beginning to all animals so that an animal can move and after some iteration stand still in a group
# - Erl.: Add repulsion and attraction force since beginning
# - Flee force leads to movement circular away from panic origin. So fine?
# - video of animals for different force paramerters and threee different panic positions
# - Add characteristics of the animals: leaders, followers, stupid, smart, etc.
# - Increase randomness to some animals so that they can suprisingly move in another different direction
# - Add third dimension to the simulation: Jumping/flying and stapling animals


import pygame
import random
import math
import numpy as np
from numba import cuda, int32
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import sys

# ------------------ Globale Terminierung ------------------
terminate_event = threading.Event()

# ------------------ Parameter der Simulation ------------------
NUM_ANIMALS = 500  # Gesamtanzahl der Tiere
NUM_CLUSTERS_IN_X = 8  # Anzahl der anfänglichen Gruppen in x-Richtung
WIDTH, HEIGHT = 150, 1000  # [dm]
ANIMAL_RADIUS = 3  # Darstellungsradius [dm]
MAX_SPEED = 0.4  # Maximale Geschwindigkeit der Tiere [dm/s]
MAX_FORCE = 15.0  # Maximale Kraft, die auf ein Tier wirken darf [kgm/s^2]
REPULSION_DIST = 10.0  # Radius um Tier, wo Abstoßungskraft wirkt. Kraft wird größer mit kleiner werdendem Abstand [dm]
REPULSION_FORCE = 0.5  # Stärke der Abstoßungskraft [kgm/s^2]
ATTRACT_DIST = 15.0  # Radius um Tier, in dem anziehende Kräfte wirken
ATTRACT_FORCE = 0.5  # Stärke der Anziehungskraft [kgm/s^2]
FLEE_FORCE = 10.0  # Fluchtkraft von der Panikquelle [kgm/s^2]
RANDOM_FORCE = 0.0  # Zufällige Kraft, die immer wirkt [kgm/s^2]
SCAL_TIME = 10.0  # Zeitfaktor für die Simulation (Skalierung der Zeit)
SHOW_FORCE_FIELDS = False  # Zeige die Kräfte an (für Debugging-Zwecke)

# Click parameters for Flee force
PANIC_CLICK_RADIUS = 100.0  # Radius, in dem ein Mausklick Panik auslöst [dm]
PANIC_BY_CLICK = True  # Panik Position durch Mausklick aktivieren
PANIC_POS_X = -5.0  # X-Position der Panikquelle (initial ungültig)
PANIC_POS_Y = -5.0  # Y-Position der Panikquelle (initial ungültig)
PANIC_TRANSMISSION_DIST = 5.0  # Abstand, in dem Panik übertragen wird [dm]

# Parameter für die Gruppenbildung beim Start
CLUSTER_SPREAD = 5  # Streuung der Tiere um das Gruppenzentrum
MARGIN = 5  # Abstand zum Fensterrand

# Aktuelle Messwerte
current_info = {
    "panic_ratio": 0.0,
    "panic_count": 0,
    "force": 0.0,
    "force_rand": 0.0,
    "force_att": 0.0,
    "force_rep": 0.0,
    "force_flee": 0.0,
}


# ------------- Globale Daten für die zeitlichen Positionen der Tiere -------------
data_pos = []  # Positionen der Tiere

# ------------- Globale Daten für den Plot -------------
data_x = []  # Simulationszeit (in s)
data_y_panic = []  # Panikverhältnis (Anteil panischer Tiere)
data_force_rand = []  # Zufallskraft
data_force_rep = []  # Abstoßungskraft
data_force_att = []  # Anziehungskraft
data_force_flee = []  # Fluchtkraft
data_force = []  # Durchschnittliche Nettokraft


# ------------------ GPU Kernel: Update der Tiere ------------------
@cuda.jit
def update_animals_kernel(
    dt,
    start_conditions,
    pos_x,
    pos_y,
    vel_x,
    vel_y,
    in_movement,
    bool_in_movement,
    panicked,
    panic_origin,
    n,
    random_generator_x,
    random_generator_y,
    forces,
    forces_random,
    forces_att,
    forces_rep,
    forces_flee,
    forces_flee_x,
    forces_flee_y,
):
    i = cuda.grid(1)

    forces[i] = 0.0
    forces_rep[i] = 0.0
    forces_att[i] = 0.0
    forces_random[i] = 0.0
    forces_flee[i] = 0.0

    if i < n:
        # Berechne alle Kräfte (ohne zu prüfen, ob die Bedingung für die Kraft erfüllt ist)
        sum_x = 0.0
        sum_y = 0.0
        count_att = 0
        count_rep = 0
        f_rep_x = 0.0
        f_rep_y = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            d_sq = dx * dx + dy * dy
            d = math.sqrt(d_sq)

            if d <= REPULSION_DIST:
                factor = REPULSION_FORCE * (d / REPULSION_DIST - 1)
                f_rep_x += dx * factor
                f_rep_y += dy * factor
                count_rep += 1

            # Anziehung zur Gruppenbildung: Berechnung Zentrum der umgebenden Tiere
            if d_sq > 0.0 and d < ATTRACT_DIST:
                sum_x += pos_x[j]
                sum_y += pos_y[j]
                count_att += 1

        # Skaliere repulsive Kräfte und speichere die repulsiven Kräfte für dieses Tier
        if count_rep > 0:
            f_rep_x /= count_rep
            f_rep_y /= count_rep
            # Berechne die resultierenden repulsiven Kräfte für dieses Tier
            forces_rep[i] = math.sqrt(f_rep_x * f_rep_x + f_rep_y * f_rep_y)

        # Anziehung zur Gruppenbildung
        if count_att > 0:
            centroid_x = sum_x / count_att
            centroid_y = sum_y / count_att
            # Vektor vom Tier zum Schwerpunkt der Gruppe der Tiere in der Nähe
            vec_x = centroid_x - pos_x[i]
            vec_y = centroid_y - pos_y[i]
            d = math.sqrt(vec_x * vec_x + vec_y * vec_y)
            if d > 0:
                factor = ATTRACT_FORCE * (1 - (2 * ANIMAL_RADIUS) / d)
                f_att_x = vec_x * factor
                f_att_y = vec_y * factor
                forces_att[i] = math.sqrt(f_att_x * f_att_x + f_att_y * f_att_y)

        # Zufällige Kraft für alle Tiere
        f_rand_x = random_generator_x[i]
        f_rand_y = random_generator_y[i]
        # Speichere die randomisierte Kräfte für dieses Tier
        forces_random[i] = math.sqrt(f_rand_x * f_rand_x + f_rand_y * f_rand_y)

        # Berechne die resultierenden Kräfte für dieses Tier
        force_x = 0.0
        force_y = 0.0
        if panicked[i] == 1:
            force_x = f_rand_x + f_att_x + f_rep_x
            force_y = f_rand_y + f_att_y + f_rep_y

        else:  # Normaler Modus ohne Panik
            # Bewegung nur für ausgewählte Tiere
            if bool_in_movement[i] == 1 and in_movement[i] == 1:
                force_x = f_rand_x + f_att_x + f_rep_x
                force_y = f_rand_y + f_att_y + f_rep_y

            else:
                force_x = 0.0
                force_y = 0.0

        # Fluchtkraft: Falls panisch und Panikquelle gültig
        if panicked[i] == 1 and panic_origin[0] >= 0:
            sign_x = 1.0 if panic_origin[0] > pos_x[i] else -1.0    # Position des Tiers relativ zur Panikquelle
            sign_y = 1.0 if panic_origin[1] > pos_y[i] else -1.0

            f_flee_x = 0.0
            f_flee_y = 0.0

            dx = pos_x[i] - ANIMAL_RADIUS - panic_origin[0]
            dy = pos_y[i] - ANIMAL_RADIUS - panic_origin[1]
            d_sq = dx * dx + dy * dy

            if d_sq > 0.0 and d_sq < PANIC_CLICK_RADIUS * PANIC_CLICK_RADIUS:
                d = math.sqrt(d_sq)
                if dx > 0:
                    dx = -dx
                if dy > 0:
                    dy = -dy
                f_flee_x = FLEE_FORCE * 0.5 * (1 - sign_x*(dx * dx) / PANIC_CLICK_RADIUS) + 0.5 * FLEE_FORCE
                f_flee_y = FLEE_FORCE * 0.5 * (1 - sign_y*(dy * dy) / PANIC_CLICK_RADIUS) + 0.5 * FLEE_FORCE
            elif d_sq == 0.0:
                f_flee_x = FLEE_FORCE
                f_flee_y = FLEE_FORCE
            elif d_sq >= PANIC_CLICK_RADIUS * PANIC_CLICK_RADIUS and forces_flee_x[i] > 0.0:        # debug: Fliehen von Tieren ausserhalb des Panikradius muss in Richtung der aktuellen Laufrichtung sein, oder panisch in irgendeine Richtung
                f_flee_x = -sign_x * 0.5 * FLEE_FORCE
                f_flee_y = -sign_y * 0.5 * FLEE_FORCE
            else:
                force_dir_x = 0.0
                force_dir_y = 0.0
                count = 0
                for j in range(n):
                    if panicked[j] == 1:
                        dx = pos_x[i] - pos_x[j] - 2 * ANIMAL_RADIUS
                        dy = pos_y[i] - pos_y[j] - 2 * ANIMAL_RADIUS
                        d_sq = dx * dx + dy * dy
                        if d_sq < PANIC_TRANSMISSION_DIST * PANIC_TRANSMISSION_DIST:
                            force_dir_x += start_conditions[4][j]* (1 - dx / PANIC_TRANSMISSION_DIST)      # Gewichtung der Laufrichtung der Tiere mit Abstand zu alle panischen Tiere, die näher als PANIC_TRANSMISSION_DIST sind: nähere panische Tiere bestimmen die Fluchtrichtung
                            force_dir_y += start_conditions[5][j]* (1 - dy / PANIC_TRANSMISSION_DIST)
                            count += 1
                if count > 0:            
                    force_dir_x /= count
                    force_dir_y /= count
                move_x = start_conditions[4][i] + force_dir_x
                move_y = start_conditions[5][i] + force_dir_y
                d_movement = math.sqrt(move_x * move_x + move_y * move_y)
                if move_x is not math.nan and move_y is not math.nan and d_movement > 0:
                    f_flee_x = move_x / d_movement * FLEE_FORCE    # Sum of direction of movement of EGO animal and of fleeing animals in its environment
                    f_flee_y = move_y / d_movement * FLEE_FORCE

            # Berechne die resultierenden Fluchtkräfte für dieses Tier
            force_x += f_flee_x
            force_y += f_flee_y
            forces_flee_x[i] = f_flee_x
            forces_flee_y[i] = f_flee_y
            # Speichere die repulsiven Kräfte für dieses Tier
            forces_flee[i] = math.sqrt(f_flee_x * f_flee_x + f_flee_y * f_flee_y)

        # Begrenze die Gesamtkraft
        f_mag = math.sqrt(force_x * force_x + force_y * force_y)
        if f_mag > MAX_FORCE:
            scale = MAX_FORCE / f_mag
            force_x *= scale  # Skaliere die Kraft in x-Richtung
            force_y *= scale  # Skaliere die Kraft in y-Richtung
            f_mag = MAX_FORCE

        # Speichere die Nettokraft (die Summe aller wirkenden Kräfte) für dieses Tier
        forces[i] = f_mag

        # Aktualisiere die Geschwindigkeit
        vel_x[i] += force_x * SCAL_TIME * dt + start_conditions[2][i]
        vel_y[i] += force_y * SCAL_TIME * dt + start_conditions[3][i]

        # Begrenze die Geschwindigkeit
        speed = math.sqrt(vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i])
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            vel_x[i] = vel_x[i] * scale
            vel_y[i] = vel_y[i] * scale

        # Aktualisiere die Position
        pos_x[i] += vel_x[i] * SCAL_TIME * dt
        pos_y[i] += vel_y[i] * SCAL_TIME * dt

        # Kollision mit den Wänden: Korrigiere Position und setze Geschwindigkeit in Richtung der Wand auf 0.
        if pos_x[i] < ANIMAL_RADIUS:
            pos_x[i] = ANIMAL_RADIUS
            vel_x[i] = 0
        elif pos_x[i] > WIDTH - ANIMAL_RADIUS:
            pos_x[i] = WIDTH - ANIMAL_RADIUS
            vel_x[i] = 0

        if pos_y[i] < ANIMAL_RADIUS:
            pos_y[i] = ANIMAL_RADIUS
            vel_y[i] = 0
        elif pos_y[i] > HEIGHT - ANIMAL_RADIUS:
            pos_y[i] = HEIGHT - ANIMAL_RADIUS
            vel_y[i] = 0


# ------------------ GPU Kernel: Panikübertragung ------------------
@cuda.jit
def transmit_panic_kernel(pos_x, pos_y, panicked, n, transmission_dist_sq):
    i = cuda.grid(1)
    if i < n:
        if panicked[i] == 0:
            for j in range(n):
                if panicked[j] == 1:
                    dx = pos_x[i] - pos_x[j] - 2 * ANIMAL_RADIUS
                    dy = pos_y[i] - pos_y[j] - 2 * ANIMAL_RADIUS
                    d_sq = dx * dx + dy * dy
                    if d_sq < transmission_dist_sq:
                        panicked[i] = 1
                        break


# ------------------ GPU Kernel: Reduktion zur Summierung des panicked-Arrays ------------------
@cuda.jit
def reduce_sum_kernel(arr, partial_sums):
    sdata = cuda.shared.array(128, dtype=int32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x * 2 + tid
    s = 0
    if i < arr.shape[0]:
        s = arr[i]
    if i + cuda.blockDim.x < arr.shape[0]:
        s += arr[i + cuda.blockDim.x]
    sdata[tid] = s
    cuda.syncthreads()
    s_val = cuda.blockDim.x // 2
    while s_val > 0:
        if tid < s_val:
            sdata[tid] += sdata[tid + s_val]
        cuda.syncthreads()
        s_val //= 2
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = sdata[0]


def gpu_reduce_sum(d_arr, n):
    threads_per_block = 128
    blocks = (n + threads_per_block * 2 - 1) // (threads_per_block * 2)
    d_partial = cuda.device_array(blocks, dtype=np.int32)
    reduce_sum_kernel[blocks, threads_per_block](d_arr, d_partial)
    cuda.synchronize()
    partial = d_partial.copy_to_host()
    return partial.sum()


# ------------------ GPU Kernel: Kollisionsauflösung ------------------
@cuda.jit
def resolve_collisions_with_force_kernel(
    pos_x, pos_y, forces, n, animal_radius, max_force
):
    """
    Für jedes Tierpaar, das sich überlappt, wird der benötigte Korrekturvektor
    berechnet. Falls der Korrekturbedarf (als angenommene "Kraft") den Spielraum des Tiers
    (max_force minus bereits wirkende Kraft aus forces[]) überschreitet, wird die
    Ortsänderung für diese Tiere auf Null gesetzt.
    """
    # Wir lassen nur den Thread mit globaler ID 0 diese Logik ausführen, da dies
    # eine sequentielle Kollisionsauflösung ist.
    if cuda.grid(1) == 0:
        min_d = (
            2 * animal_radius
        )  # minimal erlaubter Abstand zwischen den Mittelpunkten
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                d_sq = dx * dx + dy * dy
                # Wenn die Tiere sich überlappen (und nicht exakt aufeinander liegen)
                if d_sq < min_d * min_d and d_sq > 0:
                    d = math.sqrt(d_sq)
                    overlap = min_d - d
                    # Berechne die "Korrekturkraft" als halbe Verschiebung
                    required_force = overlap / 2.0
                    # Prüfe, ob die Tiere diese zusätzliche Kraft überhaupt aufbringen können,
                    # ohne ihr Maximum (max_force) zu überschreiten.
                    if (
                        forces[i] + required_force > max_force
                        or forces[j] + required_force > max_force
                    ):
                        # Falls die erforderliche Korrekturkraft zu hoch wäre,
                        # setze die Ortsänderung für diese Kollision auf 0.
                        continue
                    else:
                        # Berechne den Korrekturvektor (aufgeteilt auf beide Tiere)
                        shift_x = int(round((dx / d) * (overlap / 2)))
                        shift_y = int(round((dy / d) * (overlap / 2)))
                        pos_x[i] += shift_x
                        pos_y[i] += shift_y
                        pos_x[j] -= shift_x
                        pos_y[j] -= shift_y


# ------------------ Initialisierung der Simulation ------------------
def initialize_simulation():
    n = NUM_ANIMALS
    # Arrays für Positionen, Geschwindigkeiten, Bewegungsflag und Panikstatus
    pos_x = np.zeros(n, dtype=np.float32)
    pos_y = np.zeros(n, dtype=np.float32)
    vel_x = np.zeros(n, dtype=np.float32)
    vel_y = np.zeros(n, dtype=np.float32)
    bool_in_movement = np.zeros(n, dtype=np.int32)
    panicked = np.zeros(n, dtype=np.int32)

    # Startpositionen
    cluster_width_in_x = (WIDTH - 2 * ANIMAL_RADIUS) / NUM_CLUSTERS_IN_X
    if (
        cluster_width_in_x < 2 * ANIMAL_RADIUS
    ):  # Wenn die Clusterbreite kleiner als der Durchmesser der Tiere ist, dann nutze Durchmesser des Tieres als Clusterbreite
        num_cluster_x = int((WIDTH - 2 * ANIMAL_RADIUS) / (2 * ANIMAL_RADIUS))
    else:
        num_cluster_x = NUM_CLUSTERS_IN_X
    dx_cluster = np.linspace(ANIMAL_RADIUS, WIDTH - ANIMAL_RADIUS, num_cluster_x)
    animals_in_y = NUM_ANIMALS // num_cluster_x
    rest = NUM_ANIMALS % num_cluster_x
    if rest > 0:
        dy_cluster = np.linspace(
            ANIMAL_RADIUS, HEIGHT - ANIMAL_RADIUS, animals_in_y + 1
        )
    else:
        dy_cluster = np.linspace(ANIMAL_RADIUS, HEIGHT - ANIMAL_RADIUS, animals_in_y)

    half_cluster_width_in_x = (dx_cluster[1] - dx_cluster[0]) / 2
    half_cluster_width_in_y = (dy_cluster[1] - dy_cluster[0]) / 2

    index = 0
    for dx_pos in dx_cluster:
        for dy_pos in dy_cluster:
            if half_cluster_width_in_x - ANIMAL_RADIUS < 0:
                idx = int(dx_pos)
            else:
                idx = int(
                    dx_pos
                    + random.uniform(
                        -half_cluster_width_in_x + ANIMAL_RADIUS,
                        half_cluster_width_in_x - ANIMAL_RADIUS,
                    )
                )
            if half_cluster_width_in_y - ANIMAL_RADIUS < 0:
                idy = int(dy_pos)
            else:
                idy = int(
                    dy_pos
                    + random.uniform(
                        -half_cluster_width_in_y + ANIMAL_RADIUS,
                        half_cluster_width_in_y - ANIMAL_RADIUS,
                    )
                )
            if index < n:
                pos_x[index] = idx
                pos_y[index] = idy
                vel_x[index] = 0.0
                vel_y[index] = 0.0
                bool_in_movement[index] = 1 if random.randint(0, 1) == 1 else 0
                index += 1

    return pos_x, pos_y, vel_x, vel_y, bool_in_movement, panicked


# ------------------ Simulation: Pygame + GPU ------------------
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Massenpanik im Stall (GPU beschleunigt)")
    clock = pygame.time.Clock()
    FPS = 60
    dt = clock.tick(FPS) / 1000.0  # Zeit in Sekunden pro Frame
    dt = np.float32(dt)  # Konvertiere dt in float32 für die GPU

    pos_x, pos_y, vel_x, vel_y, in_movement, panicked = initialize_simulation()
    start_conditions = [
        pos_x,
        pos_y,
        vel_x,
        vel_y,
        np.zeros(NUM_ANIMALS, dtype=np.float32),  # Dummy-Wert für Richtungsvektor in x-Richtung
        np.zeros(NUM_ANIMALS, dtype=np.float32),  # Dummy-Wert für Richtungsvektor in y-Richtung     
    ]  # Startbedingungen für die Simulation
    data_pos.append((pos_x, pos_y))  # for plot

    n = NUM_ANIMALS

    # Kraft zu Positionsberechnung und GEschwindigkeitsberechnung
    d_start_conditions = cuda.to_device(start_conditions)

    # Übertrage den Zustand in den GPU-Speicher
    d_pos_x = cuda.to_device(pos_x)
    d_pos_y = cuda.to_device(pos_y)
    d_vel_x = cuda.to_device(vel_x)
    d_vel_y = cuda.to_device(vel_y)
    d_in_movement = cuda.to_device(in_movement)
    d_panicked = cuda.to_device(panicked)

    # Allocate array for forces (Netto-Kraft pro Tier)
    d_forces = cuda.to_device(np.empty(n, dtype=np.float32))
    d_forces_random = cuda.to_device(np.empty(n, dtype=np.float32))
    d_forces_rep = cuda.to_device(np.empty(n, dtype=np.float32))
    d_forces_att = cuda.to_device(np.empty(n, dtype=np.float32))
    d_forces_flee = cuda.to_device(np.empty(n, dtype=np.float32))

    # Array für Flee Kraft zur Panikübertragung
    d_forces_flee_x = cuda.to_device(np.zeros(n, dtype=np.float32))
    d_forces_flee_y = cuda.to_device(np.zeros(n, dtype=np.float32))

    # Panic origin: initial ungültig ([-1,-1] signalisiert "keine Panikquelle")
    panic_origin_host = np.array([-1.0, -1.0], dtype=np.float32)
    d_panic_origin = cuda.to_device(panic_origin_host)

    # random number generator
    d_forces_random_generator_x = cuda.to_device(np.empty(n, dtype=np.float32))
    d_forces_random_generator_y = cuda.to_device(np.empty(n, dtype=np.float32))
    d_bool_in_movement = cuda.to_device(np.zeros(n, dtype=np.int32))

    sim_time = 0
    while not terminate_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminate_event.set()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if PANIC_BY_CLICK:
                    click_x, click_y = event.pos
                else:
                    click_x = PANIC_POS_X
                    click_y = PANIC_POS_Y
                panic_origin_host[0] = click_x
                panic_origin_host[1] = click_y
                d_panic_origin.copy_to_device(panic_origin_host)
                # Setze Panikstatus für Tiere in der Nähe des Klicks
                pos_x_host = d_pos_x.copy_to_host()
                pos_y_host = d_pos_y.copy_to_host()
                for i in range(n):
                    dx = pos_x_host[i] - click_x
                    dy = pos_y_host[i] - click_y
                    if math.sqrt(dx * dx + dy * dy) < PANIC_CLICK_RADIUS:
                        panicked[i] = 1
                d_panicked.copy_to_device(panicked)

        threads_per_block = 128
        blocks = (n + threads_per_block - 1) // threads_per_block

        # Panikübertragung: komplett auf der GPU
        transmit_panic_kernel[blocks, threads_per_block](
            d_pos_x,
            d_pos_y,
            d_panicked,
            n,
            PANIC_TRANSMISSION_DIST * PANIC_TRANSMISSION_DIST,
        )
        cuda.synchronize()

        # Berechne das Panikverhältnis auf der GPU per Reduktion
        panicked_count = gpu_reduce_sum(d_panicked, n)
        panic_ratio = panicked_count / n

        # Speichere Werte für den Plot (sim_time in s)
        data_x.append(sim_time / 1000.0)
        data_y_panic.append(panic_ratio)

        # Berechne randomisierte Kräfte für alle Tiere
        rand_movements = np.zeros((n, 2), dtype=np.float32)
        rand_movements_bool = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rand_movements[i, 0] = (
                random.uniform(-RANDOM_FORCE, RANDOM_FORCE) if RANDOM_FORCE != 0 else 0
            )
            rand_movements[i, 1] = (
                random.uniform(-RANDOM_FORCE, RANDOM_FORCE) if RANDOM_FORCE != 0 else 0
            )
            rand_movements_bool[i] = random.randint(0, 1)
        d_forces_random_generator_x.copy_to_device(rand_movements[:, 0])
        d_forces_random_generator_y.copy_to_device(rand_movements[:, 1])

        # Ändere den Bewegunszuastand der Tiere randomisiert
        d_bool_in_movement.copy_to_device(rand_movements_bool)

        # Update der Tiere auf der GPU inkl. Berechnung der Nettokraft
        update_animals_kernel[blocks, threads_per_block](
            dt,
            d_start_conditions,
            d_pos_x,
            d_pos_y,
            d_vel_x,
            d_vel_y,
            d_in_movement,
            d_bool_in_movement,
            d_panicked,
            d_panic_origin,
            n,
            d_forces_random_generator_x,
            d_forces_random_generator_y,
            d_forces,
            d_forces_random,
            d_forces_att,
            d_forces_rep,
            d_forces_flee,
            d_forces_flee_x,
            d_forces_flee_y,
        )
        cuda.synchronize()

        pos_prev_x = start_conditions[0].copy()
        pos_prev_y = start_conditions[1].copy()

        # Update start_conditions für die nächste Iteration
        start_conditions[0] = d_pos_x.copy_to_host()
        start_conditions[1] = d_pos_y.copy_to_host()
        start_conditions[2] = d_vel_x.copy_to_host()
        start_conditions[3] = d_vel_y.copy_to_host()
        start_conditions[4] = start_conditions[0] - pos_prev_x # Bewegungsanteil in x-Richtung
        start_conditions[5] = start_conditions[1] - pos_prev_y # Bewegungsanteil in y-Richtung
        d_start_conditions = cuda.to_device(start_conditions)

        # Rufe den Kernel zur Kollisionsauflösung auf
        resolve_collisions_with_force_kernel[1, 1](
            d_pos_x, d_pos_y, d_forces, n, ANIMAL_RADIUS, MAX_FORCE
        )
        cuda.synchronize()

        # Kopiere die aktuellen Zustände zurück
        pos_x_host = start_conditions[0]
        pos_y_host = start_conditions[1]
        panicked_host = d_panicked.copy_to_host()
        forces_host = d_forces.copy_to_host()
        forces_random_host = d_forces_random.copy_to_host()
        forces_att_host = d_forces_att.copy_to_host()
        forces_rep_host = d_forces_rep.copy_to_host()
        forces_flee_host = d_forces_flee.copy_to_host()

        avg_force = float(np.mean(forces_host))
        avg_force_random = float(np.mean(forces_random_host))
        avg_force_att = float(np.mean(forces_att_host))
        avg_force_rep = float(np.mean(forces_rep_host))
        avg_force_flee = float(np.mean(forces_flee_host))

        data_force.append(avg_force)  # Durchschnittliche Nettokraft
        data_force_rand.append(avg_force_random)  # Zufallskraft
        data_force_att.append(avg_force_att)  # Anziehungskraft
        data_force_rep.append(avg_force_rep)  # Abstoßungskraft
        data_force_flee.append(avg_force_flee)  # Fluchtkraft

        # Aktualisiere die globalen Infos, die im Tkinter-Fenster angezeigt werden
        current_info["panic_ratio"] = panic_ratio
        current_info["panic_count"] = panicked_count
        current_info["force"] = avg_force
        current_info["force_rand"] = avg_force_random
        current_info["force_att"] = avg_force_att
        current_info["force_rep"] = avg_force_rep
        current_info["force_flee"] = avg_force_flee

        screen.fill((30, 30, 30))
        if SHOW_FORCE_FIELDS:
            for i in range(n):
                pygame.draw.circle(
                    screen,
                    (120, 120, 120),
                    (int(pos_x_host[i]), int(pos_y_host[i])),
                    ATTRACT_DIST,
                )
                pygame.draw.circle(
                    screen,
                    (0, 0, 255),
                    (int(pos_x_host[i]), int(pos_y_host[i])),
                    REPULSION_DIST,
                )

        for i in range(n):
            color = (255, 0, 0) if panicked_host[i] == 1 else (0, 255, 0)
            pygame.draw.circle(
                screen,
                (0, 0, 0),
                (int(pos_x_host[i]), int(pos_y_host[i])),
                ANIMAL_RADIUS,
                width=1,
            )
            pygame.draw.circle(
                screen,
                color,
                (int(pos_x_host[i]), int(pos_y_host[i])),
                ANIMAL_RADIUS - 2,
            )

        # Zeichne die Panikquelle (falls vorhanden)
        if panic_origin_host[0] >= 0:
            pygame.draw.circle(
                screen,
                (255, 0, 0),
                (int(panic_origin_host[0]), int(panic_origin_host[1])),
                PANIC_CLICK_RADIUS,
                width=1,
            )

        pygame.display.flip()

        sim_time += dt

    pygame.quit()


# ------------------ Tkinter Fenster mit Matplotlib Diagramm ------------------
def run_tkinter():
    root = tk.Tk()
    root.title("Echtzeit-Auswertung: Panikverhältnis und Kräfte")

    # Label für die aktuellen Werte
    info_label = tk.Label(root, text="", font=("Arial", 10), justify=tk.LEFT)
    info_label.pack(side=tk.TOP, anchor="w", padx=10, pady=5)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Zeit (s)")
    ax1.set_ylabel("Panikanteil", color="red")
    ax1.set_ylim(0, 1.2)
    (line1,) = ax1.plot([], [], "r-", label="Panikanteil")

    # Zweite y-Achse für die Nettokraft
    ax2 = ax1.twinx()
    ax2.set_ylabel("Durchschnittliche Nettokraft", color="blue")
    ax2.set_ylim(0, MAX_FORCE * 1.5)
    (line2,) = ax2.plot([], [], "b-", label="Kraft (gesamt)")
    (line3,) = ax2.plot([], [], "g-", label="Zufallskraft")
    (line4,) = ax2.plot([], [], "c-", label="Anziehungskraft")
    (line5,) = ax2.plot([], [], "m-", label="Abstoßungskraft")
    (line6,) = ax2.plot([], [], "y-", label="Fluchtkraft")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    def update_plot():
        if terminate_event.is_set():
            root.destroy()
            sys.exit(0)
            return
        if not (
            len(data_x)
            == len(data_y_panic)
            == len(data_force)
            == len(data_force_rand)
            == len(data_force_att)
            == len(data_force_rep)
            == len(data_force_flee)
        ):
            # Überspringe diesen Aufruf, wenn die Längen nicht übereinstimmen
            root.after(200, update_plot)
            return
        # Daten begrenzen, falls zu viele Werte vorhanden
        xdata = data_x

        line1.set_xdata(xdata)
        line1.set_ydata(data_y_panic)
        line2.set_xdata(xdata)
        line2.set_ydata(data_force)
        line3.set_xdata(xdata)
        line3.set_ydata(data_force_rand)
        line4.set_xdata(xdata)
        line4.set_ydata(data_force_att)
        line5.set_xdata(xdata)
        line5.set_ydata(data_force_rep)
        line6.set_xdata(xdata)
        line6.set_ydata(data_force_flee)

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        canvas.draw()

        # Aktualisiere das Info-Label mit den aktuellen Werten
        info_text = (
            f"Panikanteil: {current_info['panic_ratio']:.3f}\n"
            f"Panikzahl: {current_info['panic_count']}\n"
            f"Kraft (gesamt): {current_info['force']:.2f}\n"
            f"Zufallskraft: {current_info['force_rand']:.2f}\n"
            f"Anziehungskraft: {current_info['force_att']:.2f}\n"
            f"Abstoßungskraft: {current_info['force_rep']:.2f}\n"
            f"Fluchtkraft: {current_info['force_flee']:.2f}"
        )
        info_label.config(text=info_text)

        root.after(200, update_plot)  # alle 200ms aktualisieren

    def on_close():
        terminate_event.set()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_plot()
    root.mainloop()


# ------------------ Hauptprogramm ------------------
if __name__ == "__main__":
    # Starte die Simulation (Pygame + GPU) in einem separaten Thread
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    # Tkinter (mit Matplotlib) läuft im Hauptthread
    run_tkinter()
