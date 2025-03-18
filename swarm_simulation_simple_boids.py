import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class ChickenPanicSimulation:
    def __init__(self, 
                 num_chickens=100, 
                 width=100, height=100,
                 max_speed=2.0, 
                 max_acceleration=0.1,
                 neighbor_radius=10,
                 weight_separation=1.5, 
                 weight_alignment=1.0, 
                 weight_cohesion=1.0,
                 weight_goal=1.0, 
                 weight_obstacle=2.0,
                 panic_trigger_time=10.0, 
                 exit_position=np.array([90, 50]),
                 panic_origin=np.array([50, 50]),
                 panic_radius=20,
                 panic_propagation_rate=0.02,
                 panic_flee_mix=0.8  # 0: rein radial, 1: rein tangential
                ):
        # Simulationsparameter
        self.num_chickens = num_chickens
        self.width = width
        self.height = height
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.neighbor_radius = neighbor_radius
        
        self.weight_separation = weight_separation
        self.weight_alignment = weight_alignment
        self.weight_cohesion = weight_cohesion
        self.weight_goal = weight_goal
        self.weight_obstacle = weight_obstacle
        
        self.exit_position = exit_position
        self.panic_trigger_time = panic_trigger_time  # Zeitpunkt, ab dem die initiale Panik ausgelöst wird
        
        # Position, von der aus die Panik startet und zugehöriger Radius
        self.panic_origin = panic_origin
        self.panic_radius = panic_radius
        self.panic_propagation_rate = panic_propagation_rate
        self.panic_flee_mix = panic_flee_mix
        self.initial_panic_triggered = False
        
        # Initialisierung: Hühner gleichverteilt im gesamten Stall
        self.positions = np.random.rand(num_chickens, 2) * np.array([width, height])
        angles = np.random.rand(num_chickens) * 2 * np.pi
        self.velocities = np.column_stack((np.cos(angles), np.sin(angles))) * (max_speed * 0.5)
        
        # Panik-Level: 0 = ruhig, 1 = maximale Panik
        self.panic_levels = np.zeros(num_chickens)
        
        # Effekt-Toggles
        self.effect_separation = True
        self.effect_alignment = True
        self.effect_cohesion = True
        self.effect_goal = True
        self.effect_obstacle = True
        self.effect_extreme = True
        self.effect_reaction_delay = True
        
        # Zusätzliche extreme Effekte
        self.effect_stumble = True     # Stolpern
        self.effect_random = True      # Unkontrollierte Richtungswechsel
        self.effect_freeze = True      # Freezing (Erstarren)
        self.effect_blockade = True    # Blockaden an Engstellen
        
        # Individuelle Reaktionszeiten (Verzögerung in Sekunden)
        self.reaction_time = np.random.rand(num_chickens) * 1.0  
        self.current_delay = np.zeros(num_chickens)
        
        # Farben für den Plot (als NumPy-Array mit ausreichendem dtype)
        self.colors = np.array(["blue"] * self.num_chickens, dtype='<U10')
        
        # Simulationszeit und Zeitschritt
        self.time = 0.0
        self.dt = 0.1  # Zeitschritt
        
    # --- Toggle-Methoden ---
    def toggle_separation(self, active: bool):
        self.effect_separation = active
    
    def toggle_alignment(self, active: bool):
        self.effect_alignment = active
        
    def toggle_cohesion(self, active: bool):
        self.effect_cohesion = active
        
    def toggle_obstacle(self, active: bool):
        self.effect_obstacle = active
        
    def toggle_goal(self, active: bool):
        self.effect_goal = active
        
    def toggle_extreme(self, active: bool):
        self.effect_extreme = active
        
    def toggle_stumble(self, active: bool):
        self.effect_stumble = active
        
    def toggle_random(self, active: bool):
        self.effect_random = active
        
    def toggle_freeze(self, active: bool):
        self.effect_freeze = active
        
    def toggle_blockade(self, active: bool):
        self.effect_blockade = active
    
    # --- Hilfsmethode: Begrenze Vektoren auf maximale Länge ---
    def limit_vector(self, vector, max_value):
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1
        factor = np.minimum(1, max_value / norm)
        return vector * factor
    
    # --- Update-Schritt der Simulation ---
    def update(self):
        self.time += self.dt
        
        # Initialauslösung der Panik: Nur einmal, wenn die Zeit erreicht ist
        if not self.initial_panic_triggered and self.time > self.panic_trigger_time:
            self.trigger_initial_panic()
        
        # Propagiere Panik durch Nachbarschaftsinteraktion
        self.propagate_panic()
        
        # Klassische Boids-Regeln
        separation_acc = self.compute_separation() if self.effect_separation else np.zeros_like(self.velocities)
        alignment_acc  = self.compute_alignment()  if self.effect_alignment  else np.zeros_like(self.velocities)
        cohesion_acc   = self.compute_cohesion()   if self.effect_cohesion   else np.zeros_like(self.velocities)
        
        # Ziel- und Hindernisbeschleunigungen: Hier setzen wir für panische Tiere die Fluchtrichtung
        goal_acc       = self.compute_goal()       if self.effect_goal else np.zeros_like(self.velocities)
        obstacle_acc   = self.compute_obstacle()   if self.effect_obstacle else np.zeros_like(self.velocities)
        
        total_acc = (self.weight_separation * separation_acc +
                     self.weight_alignment  * alignment_acc +
                     self.weight_cohesion   * cohesion_acc +
                     self.weight_goal       * goal_acc +
                     self.weight_obstacle   * obstacle_acc)
        
        # Extreme Panik-Effekte
        if self.effect_extreme:
            total_acc += self.compute_extreme_effects()
        
        # Beschränke die Beschleunigung auf die maximale Hühner-Beschleunigung
        total_acc = self.limit_vector(total_acc, self.max_acceleration)
        
        # Neue Geschwindigkeiten berechnen
        new_velocities = self.velocities + total_acc * self.dt
        new_velocities = self.limit_vector(new_velocities, self.max_speed)
        
        # Reaktionsverzögerung: Manche Hühner übernehmen die neue Geschwindigkeit verzögert
        if self.effect_reaction_delay:
            delay_mask = self.current_delay <= 0
            self.velocities[delay_mask] = new_velocities[delay_mask]
            self.current_delay[~delay_mask] -= self.dt
        else:
            self.velocities = new_velocities
        
        # Asynchrone Positionsaktualisierung
        new_positions = self.positions + self.velocities * self.dt
        new_positions = self.handle_collisions(new_positions)
        self.positions = new_positions
        
        # Aktualisiere die Farben (basierend auf dem Panik-Level)
        self.update_colors()
        
        # Verlangsamung der Simulation für bessere Sichtbarkeit
        time.sleep(0.05)
    
    # --- Boids-Regeln ---
    def compute_separation(self):
        acc = np.zeros_like(self.positions)
        for i in range(self.num_chickens):
            diff = self.positions[i] - self.positions
            distance = np.linalg.norm(diff, axis=1)
            mask = (distance > 0) & (distance < self.neighbor_radius)
            if np.any(mask):
                repulse = np.sum(diff[mask] / distance[mask, None], axis=0)
                acc[i] = repulse
        return acc
    
    def compute_alignment(self):
        acc = np.zeros_like(self.velocities)
        for i in range(self.num_chickens):
            diff = self.positions[i] - self.positions
            distance = np.linalg.norm(diff, axis=1)
            mask = (distance > 0) & (distance < self.neighbor_radius)
            if np.any(mask):
                avg_vel = np.mean(self.velocities[mask], axis=0)
                acc[i] = avg_vel - self.velocities[i]
        return acc
    
    def compute_cohesion(self):
        acc = np.zeros_like(self.positions)
        for i in range(self.num_chickens):
            diff = self.positions[i] - self.positions
            distance = np.linalg.norm(diff, axis=1)
            mask = (distance > 0) & (distance < self.neighbor_radius)
            if np.any(mask):
                center = np.mean(self.positions[mask], axis=0)
                acc[i] = center - self.positions[i]
        return acc
    
    def compute_goal(self):
        """
        Für jedes panische Huhn wird die Fluchtrichtung berechnet. 
        Dabei wird der Vektor von der Panikquelle (panic_origin) zum Huhn ermittelt.
        Anschließend wird der Vektor rotiert (Tangente des Kreises) und mit einem
        mix-Faktor kombiniert, sodass die Fluchtrichtung aus einem radiale (weg von der Panikquelle)
        und einem tangentialen Anteil besteht.
        """
        acc = np.zeros_like(self.positions)
        for i in range(self.num_chickens):
            if self.panic_levels[i] > 0:
                # Radialvektor: vom Panikursprung zum Huhn
                radial = self.positions[i] - self.panic_origin
                norm_radial = np.linalg.norm(radial)
                if norm_radial > 0:
                    radial_norm = radial / norm_radial
                    # Tangentialvektor: Rotation um 90° (eine Richtung wird gewählt)
                    tangent = np.array([-radial_norm[1], radial_norm[0]])
                    # Mischung aus radialem und tangentialem Anteil
                    flee_direction = (1 - self.panic_flee_mix) * radial_norm + self.panic_flee_mix * tangent
                    # Normieren
                    flee_direction /= np.linalg.norm(flee_direction)
                    # Zielbeschleunigung wird mit dem Panik-Level skaliert
                    acc[i] = flee_direction * self.panic_levels[i]
            else:
                acc[i] = 0
        return acc
    
    def compute_obstacle(self):
        # Wandvermeidung: Abstoßungsvektor, skaliert mit dem Panik-Level (nur panische Tiere reagieren stark)
        acc = np.zeros_like(self.positions)
        margin = 5
        for i in range(self.num_chickens):
            steer = np.zeros(2)
            x, y = self.positions[i]
            if x < margin:
                steer[0] = 1 / (x + 1e-5)
            elif x > self.width - margin:
                steer[0] = -1 / (self.width - x + 1e-5)
            if y < margin:
                steer[1] = 1 / (y + 1e-5)
            elif y > self.height - margin:
                steer[1] = -1 / (self.height - y + 1e-5)
            acc[i] = steer * self.panic_levels[i]
        return acc
    
    def compute_extreme_effects(self):
        # Extreme Effekte: zufällige Richtungswechsel, Stolpern, Freezing
        extreme_acc = np.zeros_like(self.velocities)
        for i in range(self.num_chickens):
            if self.panic_levels[i] > 0.9:
                # Zufällige Richtungswechsel
                if self.effect_random:
                    rand_change = np.random.uniform(-1, 1, size=2)
                    norm = np.linalg.norm(rand_change)
                    if norm > 0:
                        extreme_acc[i] += (rand_change / norm) * 0.5
                # Stolpern: plötzliche Verzögerung
                if self.effect_stumble:
                    if np.random.rand() < 0.2:
                        extreme_acc[i] -= self.velocities[i]
                        self.current_delay[i] = np.random.uniform(0.5, 1.5)
                # Freezing: Erstarren
                if self.effect_freeze:
                    if np.random.rand() < 0.1:
                        extreme_acc[i] = -self.velocities[i]
                        self.current_delay[i] = np.random.uniform(2, 4)
        return extreme_acc
    
    # --- Kollisionsbehandlung (Wände und Blockaden) ---
    def handle_collisions(self, new_positions):
        # Wandkollisionen
        for i in range(self.num_chickens):
            x, y = new_positions[i]
            if x < 0:
                new_positions[i, 0] = 0
                self.velocities[i, 0] *= -0.5
            elif x > self.width:
                new_positions[i, 0] = self.width
                self.velocities[i, 0] *= -0.5
            if y < 0:
                new_positions[i, 1] = 0
                self.velocities[i, 1] *= -0.5
            elif y > self.height:
                new_positions[i, 1] = self.height
                self.velocities[i, 1] *= -0.5
        
        # Blockadeneffekt: In Bereichen hoher Dichte
        if self.effect_blockade:
            grid_size = 10
            grid_counts = {}
            indices = {}
            for i in range(self.num_chickens):
                grid_x = int(new_positions[i, 0] // grid_size)
                grid_y = int(new_positions[i, 1] // grid_size)
                key = (grid_x, grid_y)
                grid_counts[key] = grid_counts.get(key, 0) + 1
                if key not in indices:
                    indices[key] = []
                indices[key].append(i)
            for key, count in grid_counts.items():
                if count > 5:
                    for i in indices[key]:
                        new_positions[i] -= self.velocities[i] * self.dt * 0.5
        return new_positions
    
    # --- Initiale Panikauslösung ---
    def trigger_initial_panic(self):
        # Setze Panik-Level für Tiere innerhalb des definierten Radius um panic_origin
        for i in range(self.num_chickens):
            dist = np.linalg.norm(self.positions[i] - self.panic_origin)
            if dist < self.panic_radius:
                self.panic_levels[i] = 0.8  # Initial hoher Panik-Level
        self.initial_panic_triggered = True
    
    # --- Panik-Propagation ---
    def propagate_panic(self):
        # Tiere, die in der Nachbarschaft (innerhalb neighbor_radius) panische Tiere haben, erhöhen ihren Panik-Level
        for i in range(self.num_chickens):
            if self.panic_levels[i] < 1:
                diff = self.positions[i] - self.positions
                distance = np.linalg.norm(diff, axis=1)
                mask = (distance > 0) & (distance < self.neighbor_radius)
                if np.any(mask):
                    if np.any(self.panic_levels[mask] > 0.2):
                        self.panic_levels[i] = np.clip(self.panic_levels[i] + self.panic_propagation_rate, 0, 1)
    
    # --- Farben aktualisieren ---
    def update_colors(self):
        # Blau: ruhig, Orange: mittlere Panik, Rot: extreme Panik
        for i in range(self.num_chickens):
            if self.panic_levels[i] > 0.9:
                self.colors[i] = "red"
            elif self.panic_levels[i] > 0.5:
                self.colors[i] = "orange"
            else:
                self.colors[i] = "blue"
    
    # --- Hauptschleife der Simulation mit Animation ---
    def run(self, steps=500):
        fig, ax = plt.subplots(figsize=(8, 8))
        scat = ax.scatter(self.positions[:, 0], self.positions[:, 1], c=self.colors)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title("Chicken Panic Simulation")
        
        # Legende zur Markierung der Panik-Level
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Ruhig', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Mittlere Panik', markerfacecolor='orange', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Extreme Panik', markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        def animate(frame):
            self.update()
            scat.set_offsets(self.positions)
            scat.set_color(self.colors)
            return scat,
        
        ani = animation.FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)
        plt.show()

# Ausführung der Simulation
if __name__ == "__main__":
    sim = ChickenPanicSimulation(num_chickens=1500, width=15, height=100,
                                 max_speed=5.0, max_acceleration=0.1,
                                 neighbor_radius=10,
                                 weight_separation=0.0,
                                 weight_alignment=0.0,
                                 weight_cohesion=0.0,
                                 weight_goal=5.0,
                                 weight_obstacle=0.0,
                                 panic_trigger_time=2.0,
                                 exit_position=np.array([90, 50]),
                                 panic_origin=np.array([0, 0]),
                                 panic_radius=5,
                                 panic_propagation_rate=0.2,
                                 panic_flee_mix=0.8)
    sim.run(steps=1000)
