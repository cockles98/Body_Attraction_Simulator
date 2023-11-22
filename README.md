# N-Body Solver 3D
This Python code implements a numerical solution for the N-body problem in three-dimensional space using the Runge-Kutta of Order 4 (RK4) method. The N-body problem involves calculating the motion of multiple bodies under the influence of gravity.

# Problem Statement
The N-body problem refers to the gravitational interaction between multiple celestial bodies. In this context, the code addresses the motion of N bodies in three-dimensional space. The specific Ordinary Differential Equation being solved describes the gravitational interaction between these bodies. 
The ODE being solved is a set of second-order differential equations representing the gravitational interaction between N bodies. The gravitational force between two bodies is determined by Newton's law of gravitation. The acceleration of each body is influenced by the gravitational forces exerted by all other bodies in the system.

## Usage

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
* `plot_scale`: List with shape (3,2) of plot scales limits in X,Y and Z.
* The other parameters are defined to customize the animation and they're very intuitive, however, I will leave some examples below using all the available parameters.

## Code Structure
* The code begins with parameter checks to ensure the coherence of input data.
* The n_body_acceleration function calculates the acceleration suffered by each body based on masses and current positions.
* RK4 integration method is applied to numerically solve the system of ODEs and determine the motion of the bodies over time.

## Examples
`
# Initial conditions
G = 6.67430e-11             # Gravitational constant in m^3/kg/s^2
h = 86400                   # 1 day in seconds
t_max = 10 * 365 * 86400    # 10 years in seconds

# Animation
bodies_pos_list, num_frames, num_bodies = n_body_solver_3d(masses, positions, velocities, G, h, t_max)
plot_scale = np.array([[-3e11, 3e11], [-3e11, 3e11], [-3e11, 3e11]])
animation_3d(bodies_pos_list, num_frames, num_bodies, plot_scale,
             camera_angles=[90,5],
             title='Solar System (zoomed)\nView distance: 3e11 meters | Simulation time: 10 years',
             marker_sizes='auto',
             colors_list=['yellow', 'lightgray', 'gold', 'blue', 'red', 'wheat', 'lightyellow', 'lightseagreen', 'darkblue'],
             bodies_names=['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'],
             angular_rotation_speeds=[0.0,0.0],
             animation_speed=8,
             bodies_volume=masses,
             body_centered_index=0)
`
