# N-Body Solver 3D
This Python code implements a numerical solution for the N-body problem in three-dimensional space using the Runge-Kutta of Order 4 (RK4) method. The N-body problem involves calculating the motion of multiple bodies under the influence of gravity.

## Problem Statement
The N-body problem refers to the gravitational interaction between multiple celestial bodies. In this context, the code addresses the motion of N bodies in three-dimensional space. The specific ordinary differential equation (ODE) being solved describes the gravitational interaction between these bodies.

Ordinary Differential Equation (ODE)
The ODE being solved is a set of second-order differential equations representing the gravitational interaction between N bodies. The gravitational force between two bodies is determined by Newton's law of gravitation. The acceleration of each body is influenced by the gravitational forces exerted by all other bodies in the system.

Usage
python
Copy code
result_pos, num_steps, num_bodies = n_body_solver_3d(masses_list, initial_pos_list, initial_vel_list, G, h, t_max)
masses_list: List of masses of the N bodies.
initial_pos_list: List of initial positions of the N bodies in three dimensions.
initial_vel_list: List of initial velocities of the N bodies in three dimensions.
G: Gravitational constant.
h: Time step for the RK4 integration method.
t_max: Maximum simulation time.
Code Structure
The code begins with parameter checks to ensure the coherence of input data.
The n_body_acceleration function calculates the acceleration suffered by each body based on masses and current positions.
RK4 integration method is applied to numerically solve the system of ODEs and determine the motion of the bodies over time.
Requirements
Python
NumPy
How to Run
Ensure Python is installed.
Install NumPy using pip install numpy.
Run the script containing the provided n_body_solver_3d function.
License
This code is provided under the MIT License.

Feel free to use and modify the code according to your needs. If you find any issues or have suggestions for improvements, please create an issue or pull request on GitHub.
