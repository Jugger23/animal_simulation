import numpy as np
import matplotlib.pyplot as plt
import time

# Anzahl der Partikel
num_particles = 1000

# Initialisierung der Positionen (zufällig)
particles = np.random.rand(num_particles, 2) * 10  # Bereich von 0 bis 10

# Geschwindigkeiten der Partikel (zufällig)
velocities = (np.random.rand(num_particles, 2) - 0.5) * 2  # Bereich von -1 bis 1

damping_factor = 0.1  # Dämpfung beim Abprallen

# Erstellen der Plot-Figur
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter(particles[:, 0], particles[:, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Simulation starten
try:
    while True:
        particles += velocities  # Positionen aktualisieren
        
        # Begrenzungen einführen (Reflexion an den Rändern)
        for i in range(num_particles):
            for j in range(2):
                if particles[i, j] <= 0 or particles[i, j] >= 10:
                    velocities[i, j] *= -1 * damping_factor  # Richtung umkehren und dämpfen
                    particles[i, j] = np.clip(particles[i, j], 0, 10)
        
        # Kollisionsabfrage und zufälliges Abprallen mit Dämpfung
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                distance = np.linalg.norm(particles[i] - particles[j])
                if distance < 0.25:  # Annahme: Partikelradius = 0.25
                    velocities[i], velocities[j] = velocities[j] * damping_factor, velocities[i] * damping_factor
        
        # Plot aktualisieren
        sc.set_offsets(particles)
        plt.draw()
        plt.pause(0.01)  # 5 Mal pro Sekunde aktualisieren
except KeyboardInterrupt:
    print("Simulation beendet.")
