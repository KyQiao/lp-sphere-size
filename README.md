# Linear Programming (LP) Algorithm for Hard Sphere Size Measurement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains a minimum working example demonstrating the Linear Programming (LP) algorithm used to measure particle radii in 3D Random Close Packing (RCP) samples. 


## Repository Contents

* `lp_algorithm.py`: The main Python script that sets up and solves the LP problem.
* `3dRCP.csv`: A sample dataset containing 512 particles in a 3D Random Close Packing configuration. Columns represent $(x, y, z, r)$, where $x, y, z$ are the coordinates and $r$ is the currently estimated particle radius.
* `requirements.txt`: List of Python dependencies required to run the code.

## Requirements and Installation

The code relies on standard scientific Python packages (`numpy`, `scipy`, `matplotlib`) and the `cvxpy` library for convex optimization.

To install the required dependencies, we recommend creating a virtual environment and running:

```bash
pip install -r requirements.txt
```
