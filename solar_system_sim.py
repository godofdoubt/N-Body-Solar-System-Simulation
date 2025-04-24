import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pygame
import time
import math

class SolarSystemSimulation:
    def __init__(self):
        """Initialize solar system simulation"""
        # Gravitational constant (m^3 kg^-1 s^-2)
        self.G = 6.67430e-11
        
        # Scale factors to make simulation visible
        # Distance scale: 1 AU = 149.6e9 meters
        self.DISTANCE_SCALE = 1.0 / 149.6e9  # Convert meters to AU for display
        
        # Time scale: accelerate time for visualization
        self.TIME_SCALE = 86400 * 5 
        
        # Will store all celestial bodies
        self.planets = []
        self.history = []
        
        # Pygame setup
        self.width, self.height = 1000, 800
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Solar System Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        
        # Viewport parameters (for zooming and panning)
        self.view_scale = 40  # Pixels per AU
        self.view_center_x = self.width // 2
        self.view_center_y = self.height // 2
        
        # Controls
        self.paused = False
        self.show_labels = True
        self.show_orbits = True
        self.simulation_speed = 0.01  # Start with normal speed
        
        # For performance, limit how often we add points to orbits
        self.orbit_point_interval = 5
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
                # else: handle close encounters if necessary, here we just skip the term

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
                    # Calculate vector from i to j
                    r_vec = positions_reshaped[j] - positions_reshaped[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Calculate gravitational acceleration: G * M * r_hat / r^2
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
    
    def run_simulation(self, duration_days, dt_days=5.0): # Renamed inputs for clarity
        """Run the simulation for the specified duration (in days)"""
        print(f"Starting simulation with duration={duration_days} days, dt={dt_days} days")
        n_planets = len(self.planets)

        # Initial conditions (still in meters and m/s)
        initial_positions = np.array([p['position'] for p in self.planets]).flatten()
        initial_velocities = np.array([p['velocity'] for p in self.planets]).flatten()
        initial_state = np.concatenate([initial_positions, initial_velocities])

        # --- Convert time units to SECONDS for the solver ---
        seconds_per_day = 86400.0
        duration_seconds = duration_days * seconds_per_day
        dt_seconds = dt_days * seconds_per_day # Approximate step size for t_eval calculation

        # Define time span and evaluation points in SECONDS
        t_span_seconds = (0, duration_seconds)

        # Calculate the number of steps based on the desired dt_days
        num_steps = int(duration_days / dt_days) + 1
        t_eval_seconds = np.linspace(t_span_seconds[0], t_span_seconds[1], num_steps)
        # --- End of time conversion ---

        print(f"Expected number of frames: {len(t_eval_seconds)}")
        print(f"Integrating over t_span (seconds): {t_span_seconds}")

        # Solve the differential equations using SECONDS as the time variable
        solution = solve_ivp(
            self.derivative,        # Derivative function returns d(state)/d(t_seconds)
            t_span_seconds,         # Time span in seconds
            initial_state,          # Initial state in m, m/s
            method='RK45',
            t_eval=t_eval_seconds,  # Evaluate at specific points in seconds
            rtol=1e-7,              # Slightly stricter tolerance might be needed
            atol=1e-9               # Slightly stricter tolerance might be needed
        )

        # --- Add Solver Diagnostics ---
        print(f"Solver status: {solution.status}")
        if solution.status != 0:
             print(f"Solver message: {solution.message}")
        print(f"Solution shape: {solution.y.shape}")
        print(f"Number of time points computed: {len(solution.t)}")
        # --- End Diagnostics ---


        # Store times (convert back to days for history/display)
        # Use the times the solver actually returned, converted to days
        self.times = solution.t / seconds_per_day
        self.states = solution.y.T # States corresponding to solution.t

        # Reset history
        self.history = []

        # Store history for analysis and visualization
        for i, t_day in enumerate(self.times): # Iterate using the calculated times in days
            state = self.states[i]
            positions = state[:n_planets * 3].reshape(n_planets, 3) # Still in meters
            velocities = state[n_planets * 3:].reshape(n_planets, 3) # Still in m/s

             # --- Calculate Energies ---
            current_energies = self.calculate_energies(positions, velocities)

            # Store current positions (m) and velocities (m/s), time (days)
            frame_data = {
                'time': t_day,
                'positions': positions.copy(),
                'velocities': velocities.copy(),
                'energies': current_energies # Add energies to the frame data
            }
            self.history.append(frame_data)

        # Print debug info about generated frames
        print(f"Generated {len(self.history)} frames spanning approximately {duration_days} days")
        if len(self.history) > 0:
            print(f"First frame time: {self.history[0]['time']:.2f} days")
            print(f"Last frame time: {self.history[-1]['time']:.2f} days")
        else:
            print("Warning: No frames were generated!")

        # Reset planet orbits for visualization
        for planet in self.planets:
            planet['orbit'] = []

        return self.history

    def analyze_energy_conservation(self):
        """
        Analyze and plot energy conservation during the simulation
        """
        if not self.history:
             print("No history data to analyze. Run simulation first.")
             return

        times = [frame['time'] for frame in self.history] # Time in days
        kinetic = [frame['energies']['kinetic'] for frame in self.history]
        potential = [frame['energies']['potential'] for frame in self.history]
        total = [frame['energies']['total'] for frame in self.history]

        plt.figure(figsize=(12, 7))

        plt.subplot(2, 1, 1) # Plot KE and PE together
        plt.plot(times, kinetic, label='Kinetic Energy', color='orange')
        plt.plot(times, potential, label='Potential Energy', color='blue')
        plt.ylabel('Energy (Joules)')
        plt.title('System Energy Components Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2) # Plot Total Energy separately to see variation
        plt.plot(times, total, label='Total Energy', color='red')
        plt.ylabel('Total Energy (Joules)')
        plt.xlabel('Time (days)')
        # Calculate variation for y-axis limits
        if len(total) > 1:
            mean_total_energy = np.mean(total)
            energy_variation = np.max(total) - np.min(total)
            # Set tighter limits if variation is small compared to mean
            if mean_total_energy != 0 and abs(energy_variation / mean_total_energy) < 0.1: # e.g., less than 10% relative variation
                 plt.ylim(mean_total_energy - energy_variation, mean_total_energy + energy_variation)
            elif energy_variation < 1e-9: # If variation is extremely small in absolute terms
                plt.ylim(mean_total_energy - 1e-9, mean_total_energy + 1e-9)


        plt.title('Total System Energy (Conservation Check)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout() # Adjust subplot params for nice layout
        plt.savefig('energy_conservation_solar_system.png')
        print("Energy conservation plot saved as 'energy_conservation_solar_system.png'")
        plt.show()    
    

    def analyze_trajectories(self):
        """
        Plot the 3D trajectories of all planets in AU.
        """
        if not self.history:
             print("No history data to analyze. Run simulation first.")
             return

        n_planets = len(self.planets)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(n_planets):
            # Extract positions (in meters) and convert to AU for plotting
            positions_m = np.array([frame['positions'][i] for frame in self.history])
            positions_au = positions_m * self.DISTANCE_SCALE # Convert to AU

            # Get color, scaling RGB values from 0-255 to 0-1 for matplotlib
            color_rgb = self.planets[i]['color']
            color_mpl = tuple(c / 255.0 for c in color_rgb)

            ax.plot(positions_au[:, 0], positions_au[:, 1], positions_au[:, 2],
                   label=self.planets[i]['name'], color=color_mpl, linewidth=1.5)

        ax.set_xlabel('X Position (AU)')
        ax.set_ylabel('Y Position (AU)')
        ax.set_zlabel('Z Position (AU)')

        # Set equal aspect ratio for a better spatial view
        # This is tricky in 3D, but we can try to make ranges similar
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
        plt.savefig('trajectories_solar_system.png')
        print("Trajectories plot saved as 'trajectories_solar_system.png'")
        plt.show()

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates (AU) to screen coordinates (pixels)"""
        screen_x = self.view_center_x + world_x * self.view_scale
        screen_y = self.view_center_y - world_y * self.view_scale  # Invert y for screen coordinates
        return int(screen_x), int(screen_y)
    
    def visualize(self):
        """Visualize the solar system simulation"""
        if not self.history:
            print("No simulation data to visualize. Run simulation first.")
            return
        if len(self.history) <= 1:
            print("Warning: Only one frame in history. Animation will not work.")
            print("Check your simulation parameters.")
            return  # Added return to avoid trying to visualize with insufficient data

        # For interactive visualization
        frame_index = 0
        running = True
        
        max_trail_length = 100  # Reduced trail length for performance

        fractional_frame_pos = 0.0
        # Make simulation_speed represent 'simulation time units per viz loop'
        # Let's reset the default speed to 1.0 (1 sim frame per viz frame)
        self.simulation_speed = 1.0 # Adjust initial speed if needed
        
        while running and frame_index < len(self.history):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
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
                    elif event.key == pygame.K_UP:
                        self.simulation_speed *= 1.5  # Increase speed by 50%
                    elif event.key == pygame.K_DOWN:
                        self.simulation_speed /= 1.5  # Decrease speed
                    
                    elif event.key == pygame.K_1:
                        self.simulation_speed = 0.1   # Very slow (1 sim frame every 10 viz frames)
                    elif event.key == pygame.K_2:
                        self.simulation_speed = 1.0  # Normal (1 sim frame per viz frame)
                    elif event.key == pygame.K_3:
                        self.simulation_speed = 10.0  # Fast (10 sim frames per viz frame)
                    elif event.key == pygame.K_4:
                        self.simulation_speed = 100.0 # Very fast
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Mouse wheel up
                        self.view_scale *= 1.1
                    elif event.button == 5:  # Mouse wheel down
                        self.view_scale /= 1.1
            
            # Update frame index if not paused
            if not self.paused:
                # Add the desired speed to the fractional position
                fractional_frame_pos += self.simulation_speed

                # Calculate how many *whole* frames to advance
                advance_frames = int(fractional_frame_pos)
                if advance_frames > 0:
                    frame_index += advance_frames
                    # Subtract the whole frames advanced, keeping the fractional part
                    fractional_frame_pos -= advance_frames

                    # Ensure frame_index loops correctly
                    if frame_index >= len(self.history):
                        print(f"Simulation loop: Reset frame index from {frame_index} (history length: {len(self.history)})")
                        # Use modulo to handle potential large jumps and wrap around
                        frame_index = frame_index % len(self.history)
                        # Optionally reset fractional part on loop:
                        # fractional_frame_pos = 0.0
                
                #frame_index += frame_step
                #if frame_index >= len(self.history):
                 #   print(f"Reset frame index from {frame_index} to 0 (history length: {len(self.history)})")
                  #  frame_index = 0  # Loop back to start
        
            # Make sure frame_index is valid
            frame_index = min(frame_index, len(self.history) - 1)
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Get positions for current frame (these are in METERS)
            positions_m = self.history[frame_index]['positions'] # Get positions in meters
            # --- Convert positions to AU for visualization ---
            positions_au = positions_m * self.DISTANCE_SCALE
            # --- End conversion ---
            
            # Update orbits (only every few frames for performance)
            self.frame_counter += 1
            if self.frame_counter % self.orbit_point_interval == 0:
                for i, planet in enumerate(self.planets):
                    pos_au = positions_au[i] # Position in AU [x_au, y_au, z_au]
                    
                    # Store AU coordinates (e.g., x, y for 2D projection)
                    planet['orbit'].append((pos_au[0], pos_au[1])) # Store (x_au, y_au) tuple

                    # Keep trail length limited
                    if len(planet['orbit']) > max_trail_length:
                        planet['orbit'].pop(0)
                    
            
            # Draw orbits
            if self.show_orbits:
                for planet in self.planets:
                    if len(planet['orbit']) > 1:
                        # Convert the list of stored (x_au, y_au) points to screen points for drawing
                        screen_points = [self.world_to_screen(p_au[0], p_au[1]) for p_au in planet['orbit']]
                        
                        # Draw lines between consecutive screen points
                        for j in range(1, len(screen_points)):
                            alpha = int(255 * j / len(screen_points))
                            orbit_color = tuple(max(0, min(255, int(c * alpha / 255))) for c in planet['color'])
                            # --- Draw line ---
                            pygame.draw.line(self.screen, orbit_color, 
                                           screen_points[j-1], screen_points[j], 1)
            
            # Draw planets
            for i, planet in enumerate(self.planets):
                # Use the AU position for screen conversion
                pos_au = positions_au[i]
                
                # Convert to screen coordinates
                screen_x, screen_y = self.world_to_screen(pos_au[0], pos_au[1]) # Use AU coords
                
                # Scale radius for visibility
                # Radius was already scaled to AU in add_planet, maybe? Let's check.
                # No, add_planet stores scaled_radius but calculates display radius differently.
                # Let's make display radius calculation consistent based on physical radius.
                physical_radius_m = self.planets[i]['radius'] # Get original radius in meters
                physical_radius_au = physical_radius_m * self.DISTANCE_SCALE # Convert radius to AU
                
                if i == 0:  # Sun
                    display_radius = max(5, int(self.view_scale * 0.05)) # Base size + scale slightly with zoom
                else:
                    display_radius = max(2, int(self.view_scale * physical_radius_au * 50)) # Scale physical size heavily for visibility
                
                # Draw the planet
                pygame.draw.circle(self.screen, planet['color'], (screen_x, screen_y), display_radius)
                
                # Draw label if enabled
                if self.show_labels:
                    label = self.font.render(planet['name'], True, (255, 255, 255))
                    self.screen.blit(label, (screen_x + display_radius + 5, screen_y - 7))
            
            # Display time and speed
            days = self.history[frame_index]['time']
            years = days / 365.25
            time_text = f"Time: {days:.1f} days ({years:.2f} years) | Speed: {self.simulation_speed:.3f}x"
            text_surface = self.font.render(time_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
            # Display frame info
            frame_text = f"Frame: {frame_index}/{len(self.history)-1}"
            text_surface = self.font.render(frame_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 30))
            
            # Display controls
            controls = [
                "Controls:",
                "Space: Pause/Resume",
                "L: Toggle labels",
                "O: Toggle orbits",
                "+/-: Zoom in/out",
                "Up/Down: Speed up/down",
                "1-4: Speed presets",
                "ESC: Quit"
            ]
            
            for i, text in enumerate(controls):
                text_surface = self.font.render(text, True, (200, 200, 200))
                self.screen.blit(text_surface, (10, 60 + i * 20))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS
        
        pygame.quit()

# Create and run the solar system simulation
if __name__ == "__main__":
    sim = SolarSystemSimulation()
    
    # Astronomical data
    # All distances in meters
    # All velocities in m/s
    # All masses in kg
    
    # Sun data
    sun_mass = 1.989e30  # kg
    sun_radius = 696340000  # meters, scaled for visualization
    
    # Planet data - source: NASA fact sheets
    # [name, mass (kg), radius (m), 
    #  distance from sun (m), orbital velocity (m/s), color (RGB)]
    planet_data = [
        # Mercury
        ["Mercury", 3.3011e23, 2439700, 
         57.909e9, 47400, (169, 169, 169)],
        
        # Venus
        ["Venus", 4.8675e24, 6051800, 
         108.209e9, 35000, (255, 198, 73)],
        
        # Earth
        ["Earth", 5.97237e24, 6371000, 
         149.596e9, 29800, (0, 191, 255)],
        
        # Mars
        ["Mars", 6.4171e23, 3389500, 
         227.923e9, 24100, (255, 0, 0)],
        
        # Jupiter
        ["Jupiter", 1.8982e27, 69911000, 
         778.57e9, 13100, (255, 140, 0)],
        
        # Saturn
        ["Saturn", 5.6834e26, 58232000, 
         1433.53e9, 9700, (240, 230, 140)],
        
        # Uranus
        ["Uranus", 8.6810e25, 25362000, 
         2872.46e9, 6800, (173, 216, 230)],
        
        # Neptune
        ["Neptune", 1.02413e26, 24622000, 
         4495.06e9, 5400, (0, 0, 128)]
    ]
    
    # Add the Sun (at the center of the system)
    sim.add_planet("Sun", sun_mass, sun_radius, #* sim.DISTANCE_SCALE, 
                  [0, 0, 0], [0, 0, 0], (255, 255, 0))
    
    # Add planets
    for planet in planet_data:
        name, mass, radius_m, distance_m, velocity_m_s, color = planet # Use clear names
        
        # Convert to simulation units
        #scaled_radius = radius_m, #* sim.DISTANCE_SCALE
        #scaled_distance = distance_m, #* sim.DISTANCE_SCALE
        
        # Initial position (along x-axis) and velocity (along y-axis)
        position = [distance_m, 0, 0]
        velocity = [0, velocity_m_s, 0]  # Orbital velocity is perpendicular to radius
        
        sim.add_planet(name, mass, radius_m, position, velocity, color) # Pass radius in meters
    
    # Run the simulation for 5 Earth years with larger time steps
    print("Running simulation...")
    sim.run_simulation(duration_days=365.25 * 1, dt_days=1.0)  # 5 years with 5-day steps
    print("Simulation complete. Starting visualization...")
    
    # Visualize the results (optional, can be commented out if only analysis is needed)
    print("Starting visualization...")
    try:
        sim.visualize()
    except pygame.error as e:
         print(f"Pygame visualization closed or encountered an error: {e}")
    print("Visualization finished.")

    # Visualize the results
    print("Starting analysis...")
    sim.analyze_energy_conservation()
    sim.analyze_trajectories()
    print("Analysis complete.")
