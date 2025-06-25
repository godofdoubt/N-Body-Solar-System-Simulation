import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pygame
import time
import math
import config  # Import the new configuration file

class SolarSystemSimulation:
    def __init__(self):
        """Initialize solar system simulation using settings from config.py"""
        # Gravitational constant from config
        self.G = config.G
        
        # Scale factor for converting meters to AU for display
        self.DISTANCE_SCALE = config.DISTANCE_SCALE
        
        # Will store all celestial bodies
        self.planets = []
        self.history = []
        
        # Pygame setup from config
        pygame.init()
        self.width, self.height = config.WIDTH, config.HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Solar System Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE)
        
        # Viewport parameters (for zooming and panning) from config
        self.view_scale = config.VIEW_SCALE_INITIAL
        self.view_center_x = self.width // 2
        self.view_center_y = self.height // 2
        
        # Controls
        self.paused = False
        self.show_labels = True
        self.show_orbits = True
        self.simulation_speed = 1.0 # Start with normal speed
        
        # For performance, limit how often we add points to orbits (from config)
        self.orbit_point_interval = config.ORBIT_POINT_INTERVAL
        self.frame_counter = 0

    def calculate_energies(self, positions, velocities):
        """
        Calculate kinetic, potential, and total energy for the system
        Assumes positions are in meters and velocities are in m/s
        """
        n_planets = len(self.planets)
        kinetic_energy = 0
        potential_energy = 0

        # Ensure positions and velocities are reshaped correctly if passed flat
        positions_reshaped = positions.reshape(n_planets, 3)
        velocities_reshaped = velocities.reshape(n_planets, 3)

        # Calculate kinetic energy
        for i in range(n_planets):
            kinetic_energy += 0.5 * self.planets[i]['mass'] * np.sum(velocities_reshaped[i]**2)

        # Calculate potential energy (sum over unique pairs)
        for i in range(n_planets):
            for j in range(i + 1, n_planets): # Avoid double counting and self-interaction
                r_vec = positions_reshaped[j] - positions_reshaped[i]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-6: # Avoid division by zero if planets get extremely close
                     potential_energy -= self.G * self.planets[i]['mass'] * self.planets[j]['mass'] / r_mag

        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy
        }    
    
    def add_planet(self, name, mass, radius, position, velocity, color):
        """Add a planet to the simulation"""
        self.planets.append({
            'name': name,
            'mass': mass, #kg
            'radius': radius,  # Store original radius in METERS
            'position': np.array(position, dtype=float), # meters
            'velocity': np.array(velocity, dtype=float), # m/s
            'color': color,
            'orbit': []  # Will store orbital path
        })
    
    def calculate_acceleration(self, positions):
        """Calculate acceleration for all bodies based on gravitational forces"""
        n_planets = len(self.planets)
        positions_reshaped = positions.reshape(n_planets, 3)
        accelerations = np.zeros_like(positions_reshaped)
        
        for i in range(n_planets):
            for j in range(n_planets):
                if i != j:
                    r_vec = positions_reshaped[j] - positions_reshaped[i]
                    r_mag = np.linalg.norm(r_vec)
                    acceleration = self.G * self.planets[j]['mass'] * r_vec / (r_mag**3)
                    accelerations[i] += acceleration
        
        return accelerations.flatten()
    
    def derivative(self, t, y):
        """System of differential equations for the N-body problem"""
        n_planets = len(self.planets)
        positions = y[:n_planets * 3]
        velocities = y[n_planets * 3:]
        
        accelerations = self.calculate_acceleration(positions)
        
        return np.concatenate([velocities, accelerations])
    
    def run_simulation(self, duration_days, dt_days):
        """Run the simulation for the specified duration (in days)"""
        print(f"Starting simulation with duration={duration_days} days, dt={dt_days} days")
        n_planets = len(self.planets)

        initial_positions = np.array([p['position'] for p in self.planets]).flatten()
        initial_velocities = np.array([p['velocity'] for p in self.planets]).flatten()
        initial_state = np.concatenate([initial_positions, initial_velocities])

        seconds_per_day = 86400.0
        duration_seconds = duration_days * seconds_per_day
        t_span_seconds = (0, duration_seconds)
        
        num_steps = int(duration_days / dt_days) + 1
        t_eval_seconds = np.linspace(t_span_seconds[0], t_span_seconds[1], num_steps)

        print(f"Expected number of frames: {len(t_eval_seconds)}")
        print(f"Integrating over t_span (seconds): {t_span_seconds}")

        solution = solve_ivp(
            self.derivative, t_span_seconds, initial_state,
            method='RK45', t_eval=t_eval_seconds, rtol=1e-7, atol=1e-9
        )

        print(f"Solver status: {solution.status}")
        if solution.status != 0:
             print(f"Solver message: {solution.message}")
        print(f"Solution shape: {solution.y.shape}")
        print(f"Number of time points computed: {len(solution.t)}")

        self.times = solution.t / seconds_per_day
        self.states = solution.y.T
        self.history = []

        for i, t_day in enumerate(self.times):
            state = self.states[i]
            positions = state[:n_planets * 3].reshape(n_planets, 3)
            velocities = state[n_planets * 3:].reshape(n_planets, 3)
            current_energies = self.calculate_energies(positions, velocities)
            frame_data = {
                'time': t_day,
                'positions': positions.copy(),
                'velocities': velocities.copy(),
                'energies': current_energies
            }
            self.history.append(frame_data)

        print(f"Generated {len(self.history)} frames spanning approximately {duration_days} days")
        if len(self.history) > 0:
            print(f"First frame time: {self.history[0]['time']:.2f} days")
            print(f"Last frame time: {self.history[-1]['time']:.2f} days")
        else:
            print("Warning: No frames were generated!")

        for planet in self.planets:
            planet['orbit'] = []

        return self.history

    def analyze_energy_conservation(self):
        """Analyze and plot energy conservation during the simulation"""
        if not self.history:
             print("No history data to analyze. Run simulation first.")
             return

        times = [frame['time'] for frame in self.history]
        kinetic = [frame['energies']['kinetic'] for frame in self.history]
        potential = [frame['energies']['potential'] for frame in self.history]
        total = [frame['energies']['total'] for frame in self.history]

        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(times, kinetic, label='Kinetic Energy', color='orange')
        plt.plot(times, potential, label='Potential Energy', color='blue')
        plt.ylabel('Energy (Joules)')
        plt.title('System Energy Components Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(times, total, label='Total Energy', color='red')
        plt.ylabel('Total Energy (Joules)')
        plt.xlabel('Time (days)')
        if len(total) > 1:
            mean_total_energy = np.mean(total)
            energy_variation = np.max(total) - np.min(total)
            if mean_total_energy != 0 and abs(energy_variation / mean_total_energy) < 0.1:
                 plt.ylim(mean_total_energy - energy_variation, mean_total_energy + energy_variation)
            elif energy_variation < 1e-9:
                plt.ylim(mean_total_energy - 1e-9, mean_total_energy + 1e-9)

        plt.title('Total System Energy (Conservation Check)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Use filename from config
        plt.savefig(config.ENERGY_PLOT_FILENAME)
        print(f"Energy conservation plot saved as '{config.ENERGY_PLOT_FILENAME}'")
        plt.show()    
    
    def analyze_trajectories(self):
        """Plot the 3D trajectories of all planets in AU."""
        if not self.history:
             print("No history data to analyze. Run simulation first.")
             return

        n_planets = len(self.planets)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(n_planets):
            positions_m = np.array([frame['positions'][i] for frame in self.history])
            positions_au = positions_m * self.DISTANCE_SCALE
            color_rgb = self.planets[i]['color']
            color_mpl = tuple(c / 255.0 for c in color_rgb)
            ax.plot(positions_au[:, 0], positions_au[:, 1], positions_au[:, 2],
                   label=self.planets[i]['name'], color=color_mpl, linewidth=1.5)

        ax.set_xlabel('X Position (AU)')
        ax.set_ylabel('Y Position (AU)')
        ax.set_zlabel('Z Position (AU)')
        
        all_pos_au = np.vstack([frame['positions'] for frame in self.history]) * self.DISTANCE_SCALE
        max_range = np.array([all_pos_au[:,0].max()-all_pos_au[:,0].min(),
                              all_pos_au[:,1].max()-all_pos_au[:,1].min(),
                              all_pos_au[:,2].max()-all_pos_au[:,2].min()]).max() / 2.0
        mid_x = (all_pos_au[:,0].max()+all_pos_au[:,0].min()) * 0.5
        mid_y = (all_pos_au[:,1].max()+all_pos_au[:,1].min()) * 0.5
        mid_z = (all_pos_au[:,2].max()+all_pos_au[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.title('Planet Trajectories')
        plt.legend()
        # Use filename from config
        plt.savefig(config.TRAJECTORY_PLOT_FILENAME)
        print(f"Trajectories plot saved as '{config.TRAJECTORY_PLOT_FILENAME}'")
        plt.show()

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates (AU) to screen coordinates (pixels)"""
        screen_x = self.view_center_x + world_x * self.view_scale
        screen_y = self.view_center_y - world_y * self.view_scale
        return int(screen_x), int(screen_y)
    
    def visualize(self):
        """Visualize the solar system simulation"""
        if not self.history:
            print("No simulation data to visualize. Run simulation first.")
            return
        if len(self.history) <= 1:
            print("Warning: Only one frame in history. Animation will not work.")
            print("Check your simulation parameters.")
            return

        frame_index = 0
        running = True
        # Use settings from config
        max_trail_length = config.MAX_TRAIL_LENGTH
        pan_speed = config.PAN_SPEED
        fractional_frame_pos = 0.0
        
        # State for mouse dragging
        dragging = False
        last_mouse_pos = None

        while running and frame_index < len(self.history):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_mouse_pos = event.pos
                    elif event.button == 4:
                        self.view_scale *= 1.1
                    elif event.button == 5:
                        self.view_scale /= 1.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                        last_mouse_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        current_pos = event.pos
                        dx = current_pos[0] - last_mouse_pos[0]
                        dy = current_pos[1] - last_mouse_pos[1]
                        self.view_center_x += dx
                        self.view_center_y += dy
                        last_mouse_pos = current_pos
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_l:
                        self.show_labels = not self.show_labels
                    elif event.key == pygame.K_o:
                        self.show_orbits = not self.show_orbits
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.view_scale *= 1.1
                    elif event.key == pygame.K_MINUS:
                        self.view_scale /= 1.1
                    elif event.key == pygame.K_1: self.simulation_speed = 0.1
                    elif event.key == pygame.K_2: self.simulation_speed = 1.0
                    elif event.key == pygame.K_3: self.simulation_speed = 10.0
                    elif event.key == pygame.K_4: self.simulation_speed = 100.0

            # Handle continuous key presses for panning
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.view_center_x -= pan_speed
            if keys[pygame.K_RIGHT]:
                self.view_center_x += pan_speed
            if keys[pygame.K_UP]:
                self.view_center_y -= pan_speed
            if keys[pygame.K_DOWN]:
                self.view_center_y += pan_speed
            
            if not self.paused:
                fractional_frame_pos += self.simulation_speed
                advance_frames = int(fractional_frame_pos)
                if advance_frames > 0:
                    frame_index += advance_frames
                    fractional_frame_pos -= advance_frames
                    if frame_index >= len(self.history):
                        frame_index = frame_index % len(self.history)
            
            frame_index = min(frame_index, len(self.history) - 1)
            
            self.screen.fill((0, 0, 0))
            
            positions_m = self.history[frame_index]['positions']
            positions_au = positions_m * self.DISTANCE_SCALE
            
            self.frame_counter += 1
            if self.frame_counter % self.orbit_point_interval == 0:
                for i, planet in enumerate(self.planets):
                    pos_au = positions_au[i]
                    planet['orbit'].append((pos_au[0], pos_au[1]))
                    if len(planet['orbit']) > max_trail_length:
                        planet['orbit'].pop(0)
            
            if self.show_orbits:
                for planet in self.planets:
                    if len(planet['orbit']) > 1:
                        screen_points = [self.world_to_screen(p_au[0], p_au[1]) for p_au in planet['orbit']]
                        for j in range(1, len(screen_points)):
                            alpha = int(255 * j / len(screen_points))
                            orbit_color = tuple(max(0, min(255, int(c * alpha / 255))) for c in planet['color'])
                            pygame.draw.line(self.screen, orbit_color, 
                                           screen_points[j-1], screen_points[j], 1)
            
            for i, planet in enumerate(self.planets):
                pos_au = positions_au[i]
                screen_x, screen_y = self.world_to_screen(pos_au[0], pos_au[1])
                
                physical_radius_m = self.planets[i]['radius']
                physical_radius_au = physical_radius_m * self.DISTANCE_SCALE
                
                if i == 0: # Sun
                    display_radius = max(5, int(self.view_scale * 0.05))
                else:
                    display_radius = max(2, int(self.view_scale * physical_radius_au * 50))
                
                pygame.draw.circle(self.screen, planet['color'], (screen_x, screen_y), display_radius)
                
                if self.show_labels:
                    label = self.font.render(planet['name'], True, (255, 255, 255))
                    self.screen.blit(label, (screen_x + display_radius + 5, screen_y - 7))
            
            days = self.history[frame_index]['time']
            years = days / 365.25
            time_text = f"Time: {days:.1f} days ({years:.2f} years) | Speed: {self.simulation_speed:.3f}x"
            text_surface = self.font.render(time_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
            frame_text = f"Frame: {frame_index}/{len(self.history)-1}"
            text_surface = self.font.render(frame_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 30))
            
            # Use controls list from config
            for i, text in enumerate(config.CONTROLS_TEXT):
                text_surface = self.font.render(text, True, (200, 200, 200))
                self.screen.blit(text_surface, (self.width - 180, 10 + i * 20))
            
            pygame.display.flip()
            # Use FPS from config
            self.clock.tick(config.FPS)
        
        pygame.quit()

# Create and run the solar system simulation using config
if __name__ == "__main__":
    sim = SolarSystemSimulation()
    
    # Add Sun from config
    sun_name, sun_mass, sun_radius, sun_pos, sun_vel, sun_color = config.SUN_DATA
    sim.add_planet(sun_name, sun_mass, sun_radius, sun_pos, sun_vel, sun_color)
    
    # Add planets from config
    for planet_data in config.PLANET_DATA:
        name, mass, radius_m, distance_m, velocity_m_s, color = planet_data
        position = [distance_m, 0, 0]
        velocity = [0, velocity_m_s, 0]
        sim.add_planet(name, mass, radius_m, position, velocity, color)
    
    print("Running simulation...")
    # Run simulation with parameters from config
    sim.run_simulation(duration_days=config.DURATION_DAYS, dt_days=config.DT_DAYS)
    print("Simulation complete.")
    
    print("Starting visualization...")
    try:
        sim.visualize()
    except pygame.error as e:
         print(f"Pygame visualization closed or encountered an error: {e}")
    print("Visualization finished.")

    print("Starting analysis...")
    sim.analyze_energy_conservation()
    sim.analyze_trajectories()
    print("Analysis complete.")
