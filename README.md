# N-Body Solar System Simulation

This project is an N-body simulation that models the gravitational interactions of celestial bodies, specifically a simplified solar system. It utilizes `scipy.integrate.solve_ivp` for numerical integration of the equations of motion and `pygame` for real-time visualization.

## Features

* **N-Body Gravity:** Simulates gravitational forces between multiple bodies simultaneously using Newton's Law of Universal Gravitation.
* **Numerical Integration:** Employs `scipy.integrate.solve_ivp` with an adaptive step-size Runge-Kutta method (RK45) for trajectory calculation.
* **Real-time Visualization:** Uses `pygame` to render the positions and trajectories of the celestial bodies.
* **Energy Conservation Analysis:** Includes functionality to calculate and optionally display the total kinetic and potential energy of the system, providing a check on the simulation's accuracy and the numerical method's stability.
* **Configurable System:** Allows defining planets (or other bodies) with their mass, initial position, and initial velocity.

## Installation

To run this simulation, you need to have Python installed. You can then install the necessary libraries using `pip`:

```bash
pip install numpy matplotlib scipy pygame
```

## Usage

1.  Save the provided Python code as a `.py` file (e.g., `solar_system_sim.py`).
2.  Open a terminal or command prompt.
3.  Navigate to the directory where you saved the file.
4.  Run the simulation using the command:

```bash
python solar_system_sim.py
```

The `pygame` window should open (it might take some time initially while the simulation calculates the trajectory, please be patient), displaying the simulation. The simulation runs for a predefined duration. After closing the `pygame` window, it may display energy conservation and trajectory plots using `matplotlib`.

## Code Structure

The simulation is encapsulated within the `SolarSystemSimulation` class. Key components include:

* **`__init__`:** Initializes simulation parameters, constants, celestial bodies, and Pygame settings.
* **`calculate_acceleration`:** Computes the gravitational acceleration on each body due to all other bodies.
* **`derivative`:** Defines the system of first-order ordinary differential equations (ODEs) for positions and velocities, used by `solve_ivp`.
* **`calculate_energies`:** Calculates the total kinetic and potential energy of the system.
* **`run_simulation`:** Pre-calculates the trajectory using `solve_ivp` and stores the results in the `history` attribute.
* **`visualize`:** Manages the `pygame` visualization loop, drawing the system state based on the pre-calculated results from `history`.
* **`world_to_screen`:** Converts simulation coordinates (AU) to screen coordinates (pixels) for visualization.
* **`analyze_energy_conservation`:** Plots kinetic, potential, and total energy over time using `matplotlib`.
* **`analyze_trajectories`:** Plots the 3D trajectories of the celestial bodies in AU using `matplotlib`.

## Scientific Accuracy and Math

This simulation is based on **Newtonian Gravity** and demonstrates its principles:

* **Newton's Law of Universal Gravitation:** The `calculate_acceleration` function implements the inverse square law of gravity.
* **Newton's Second Law (F=ma):** The calculated accelerations determine the change in velocity and position over time, as integrated by `solve_ivp`.
* **Orbital Mechanics:** The simulation visualizes orbital motion, demonstrating concepts like planets orbiting a central mass.
* **Kepler's Laws (Qualitatively):** While often using simplified initial conditions, the simulation qualitatively shows aspects like differing orbital periods based on distance.
* **N-Body Interactions:** The simulation accounts for the gravitational influence of *all* bodies on each other.
* **Energy Conservation:** The energy analysis tools allow checking how well the numerical method conserves the total energy of the system, a fundamental principle in a closed gravitational system.

**Mathematical Foundation:**

* The simulation solves a system of coupled second-order differential equations derived from Newton's laws, converted into a system of first-order ODEs suitable for `solve_ivp`.
* Acceleration calculation uses vector math and the inverse square law.
* Energy calculations use standard formulas for kinetic ($1/2 mv^2$) and gravitational potential energy ($-G m_1 m_2 / r$).

**Simplifications:**

It's important to note that this simulation is a **Newtonian** model and includes common simplifications:

* **No General Relativity:** Does not account for relativistic effects like the precession of Mercury's perihelion.
* **Point Masses:** Treats celestial bodies as points, ignoring size, shape, rotation, and internal structure.
* **No Non-Gravitational Forces:** Excludes effects like atmospheric drag, solar wind, radiation pressure, or the Yarkovsky effect.
* **Instantaneous Gravity:** Assumes gravity acts instantaneously across distance.
* **Simplified Initial Conditions:** Often starts with simplified orbits (e.g., initially circular and co-planar) rather than using precise, real-world state vectors from sources like JPL HORIZONS.

Despite these simplifications, the simulation serves as a useful tool for understanding and visualizing the fundamental principles of classical gravitational mechanics.

Special Thanks to
Dot Physics Youtube Channel , Gemini 2.5 and Claude 3.7 .
