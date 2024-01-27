# N-Body Solver 3D
The N-body problem involves calculating the motion of multiple bodies under the influence of each other (gravity). This Python code implements a numerical solution for the N-body problem in 3D space using the Runge-Kutta of Order 4 (RK4) method, and then creates an animation to display it. 

# Problem Statement
The N-body problem refers to the gravitational interaction between multiple bodies, and the specific Ordinary Differential Equation (ODE) being solved describes the gravitational interaction between these bodies. More precisely, the ODE being solved is a set of second-order ODE representing the gravitational interaction between all bodies. 
OBS: The gravitational force between two bodies is determined by Newton's law of gravitation, and the acceleration of each body is influenced by the gravitational forces exerted by all other bodies in the system.

# Usage
```python
bodies_pos_list, num_frames, num_bodies = n_body_solver_3d(masses_list, initial_pos_list, initial_vel_list, G, h, t_max)
animation_3d(bodies_pos_list, num_frames, num_bodies, plot_scale)
```

n_body_solver_3d:
* `masses_list`: List of masses of the N bodies.
* `initial_pos_list`: List of initial positions of the N bodies in three dimensions.
* `initial_vel_list`: List of initial velocities of the N bodies in three dimensions.
* `G`: Gravitational constant.
* `h`: Time step for the RK4 integration method.
* `t_max`: Maximum simulation time.

animation_3d:
* `bodies_pos_list`: List of bodies positions, returned by n_body_solver.
* `num_frames`: Number of frames in animation (number of steps used in n_body_solver).
* `num_bodies`: Number of bodies in the system.
* `plot_scale`: List with (3,2) format of plot scale limits in X,Y and Z.
* The other parameters are defined to customize the animation and they're very intuitive, however, I will leave some examples below using all the available parameters.

# Code Structure
n_body_solver_3d:
* The code begins with parameter checks to ensure the coherence of input data.
* The n_body_acceleration function calculates the acceleration suffered by each body based on masses and current positions.
* RK4 integration method is applied to numerically solve the system of ODEs and determine the motion of the bodies over time.

animation_3d:
* The code begins with parameter checks to ensure the coherence of input data.
* The view_size function calculates the marker size of all bodies, based on their volume.
* The camera_pos function calculate the current coordenates of the camera, based on it's view angles and the plot scale.
