# Particle Simulation - Maxwell-Boltzman Distribution

This Python script simulates a 2D gas of spherical particles with elastic collisions and visualizes their velocity distribution using the Maxwell-Boltzmann distribution.

Note: Much of the code is in Portuguese because it was used as a learning tool in Portugal

## Requirements:
- `numpy`
- `scipy`
- `matplotlib`

## Overview:
- **MDSimulação Class**: Simulates a system of `n` particles, each with a given position and velocity. Particles collide with each other and with the walls, and the velocities are updated accordingly. 
- **Velocity Histogram**: The velocity distribution is plotted and updated in real-time during the simulation.
- **Maxwell-Boltzmann Distribution**: The simulation compares the observed velocity distribution to the theoretical Maxwell-Boltzmann distribution.

## Usage:
1. Input the number of particles (`n`), gas type (O, C, H, or Ar), temperature (`T`), and frames per second (`FPS`) for the animation.
2. The simulation runs with elastic collisions and updates the particle positions and velocities over time.
3. The animation updates the position of the particles and plots the evolving velocity distribution.
