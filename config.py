# config.py
# This file contains all the settings and data for the solar system simulation.

# --- PHYSICAL CONSTANTS ---
# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67430e-11
# Astronomical Unit in meters, used for scaling
AU = 149.6e9


# --- SIMULATION PARAMETERS ---
# Duration of the simulation in Earth days
DURATION_DAYS = 365.25 * 5  # 5 years
# Time step for the solver in Earth days. Smaller values are more accurate but slower.
DT_DAYS = 1.0


# --- VISUALIZATION & UI SETTINGS ---
# Screen dimensions
WIDTH, HEIGHT = 1000, 800
# Frames per second for the visualization
FPS = 60

# Initial viewport settings
VIEW_SCALE_INITIAL = 40  # Pixels per AU
# Panning speed with arrow keys (pixels per frame)
PAN_SPEED = 10

# Font settings
FONT_NAME = 'Arial'
FONT_SIZE = 14

# Orbit trail settings
MAX_TRAIL_LENGTH = 100
# Update orbit trail every N frames (for performance)
ORBIT_POINT_INTERVAL = 5


# --- CELESTIAL BODY DATA ---
# Format: [Name, Mass (kg), Radius (m), Initial Distance from Sun (m), Initial Velocity (m/s), Color (R, G, B)]
SUN_DATA = ["Sun", 1.989e30, 696340000, [0, 0, 0], [0, 0, 0], (255, 255, 0)]

PLANET_DATA = [
    ["Mercury", 3.3011e23, 2439700, 57.909e9, 47400, (169, 169, 169)],
    ["Venus", 4.8675e24, 6051800, 108.209e9, 35000, (255, 198, 73)],
    ["Earth", 5.97237e24, 6371000, 149.596e9, 29800, (0, 191, 255)],
    ["Mars", 6.4171e23, 3389500, 227.923e9, 24100, (255, 0, 0)],
    ["Jupiter", 1.8982e27, 69911000, 778.57e9, 13100, (255, 140, 0)],
    ["Saturn", 5.6834e26, 58232000, 1433.53e9, 9700, (240, 230, 140)],
    ["Uranus", 8.6810e25, 25362000, 2872.46e9, 6800, (173, 216, 230)],
    ["Neptune", 1.02413e26, 24622000, 4495.06e9, 5400, (0, 0, 128)]
]


# --- UI TEXT ---
# Text for the on-screen controls display
CONTROLS_TEXT = [
    "Controls:",
    "Space: Pause/Resume",
    "Arrows: Pan view",
    "Mouse Drag: Pan view",
    "+/- or Scroll: Zoom in/out",
    "L: Toggle labels",
    "O: Toggle orbits",
    "1-4: Speed presets",
    "ESC: Quit"
]


# --- SCALING FACTORS (Derived from other constants) ---
# Converts meters to Astronomical Units (AU) for display
DISTANCE_SCALE = 1.0 / AU


# --- ANALYSIS SETTINGS ---
# Filenames for saved plots
ENERGY_PLOT_FILENAME = 'energy_conservation_solar_system.png'
TRAJECTORY_PLOT_FILENAME = 'trajectories_solar_system.png'
