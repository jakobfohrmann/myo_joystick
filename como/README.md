# Joystick Simulation

## Overview

This project contains a MuJoCo simulation setup for a hand-controller interaction system.

## Central Files

The two central files in this project are:

1. **`project/hand_in_pose.py`** - Main Python script that loads and visualizes the hand-controller model, sets muscle activations, and runs the simulation.

2. **`project/xml/controller_with_hand.xml`** - Main MuJoCo XML model file that defines the hand, controller, and thumbstick system with all their interactions, constraints, and visual elements.

## Setup

### Cloning the Repository

Important when cloning this repository, you need to initialize and update the Git submodules:

```bash
git clone <main-repo-url>
cd trainingtest
git submodule update --init --recursive
```

This will ensure that all submodule dependencies (including `myo_sim_repo/myo_sim`) are properly initialized and their contents are available.



