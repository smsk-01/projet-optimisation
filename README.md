# Convex Optimisation - L'enclos à moutons (The Sheep Enclosure)

## Project Overview

This project, part of the Optimization course at **MINES Paris-PSL (2022-2023)**, focuses on determining the optimal shape for a sheep enclosure to maximize its surface area. The enclosure is built using a fence of a fixed length (L), attached at both ends to a straight barn.

The fence is modeled as a continuous curve in the 2D plane, discretized into N+1 points $(x_i, y_i)$. The objective is to find the curve shape that maximizes the enclosed area between the fence and the barn (represented by the x-axis).

## Mathematical Formulation

The problem is formulated as an optimization task:

1.  **Objective:** Maximize the area under the curve (or minimize the negative area). This is expressed as a linear objective function $c^T z$ where z includes the decision variables (e.g., y-coordinates $y_i$ and segment slopes $t_i = \tan \theta_i$).
2.  **Constraints:**
    * Equality constraints $g(z)=0$ enforce the fixed total length of the fence and the connection between consecutive points $(x_i, y_i)$ and $(x_{i+1}, y_{i+1})$ based on the step lengths $d_i = x_{i+1} - x_i$ and slopes $t_i$.
    * Boundary conditions fix the start and end points of the fence to the barn at $(a, 0)$ and $(b, 0)$.
    * (Optional) Inequality constraints $h(z) \le 0$ are introduced later to account for a river limiting the available field depth ($y_{max}$ at a given x).

## Numerical Resolution

The project involves implementing numerical algorithms in **Python** to solve the formulated optimization problems.

### Core Tasks:

1.  **`optimal_curve` Function:** Develop an algorithm to find the optimal fence shape $(x_{opt}, y_{opt}, t_{opt})$ for the basic problem without the river constraint. The function takes the barn endpoints (a, b), fence length (L), number of segments (N), step distribution (d), and an initial guess as input. Uniform discretization is used if `d` is not provided.
2.  **Analysis:** Test the function with varying parameters (L, N), analyze the geometric nature of the solution, and study the algorithm's execution time and complexity as a function of N.
3.  **River Constraint:**
    * Implement a function `y_max_func` to interpolate the river boundary ($y_{max}$) based on satellite data provided in a `.csv` file.
    * Develop `optimal_bounded_curve`, a new algorithm incorporating the river inequality constraint $h(z) \le 0$.
4.  **(Optional) Further Extensions:**
    * Implement adaptive step size refinement based on curve curvature.
    * Discuss limitations of the current model (e.g., inability to handle "overhanging" curves) and propose alternative formulations.

## Files in this Repository (Expected)

* `main_script.py`: (Or similar name) Contains the Python implementation of the optimization algorithms (`optimal_curve`, `optimal_bounded_curve`, `y_max_func`, etc.).
* `data.csv`: (Or similar name) Contains the satellite data for the river boundary.
* `README.md`: This file.
* `(Optional)` A report (`.pdf`, notebook, etc.) detailing the analysis, results, and graphs.

## How to Run

1.  **Prerequisites:** Ensure you have Python installed along with necessary libraries (e.g., NumPy, SciPy for numerical operations and optimization).
2.  **Data:** Place the `data.csv` file in the same directory as the script.
3.  **Execution:** Run the main Python script. Modify parameters within the script or via command-line arguments (if implemented) to explore different scenarios.
