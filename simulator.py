%matplotlib notebook
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import itertools

# Just to ignore the warning in 'dist' parameter in animation functions
# (the library is warning the parameter will be removed)
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning

# Increase the max disc use, just for sure
plt.rcParams['animation.embed_limit'] = 60000


def n_body_solver_3d(masses_list, initial_pos_list, initial_vel_list, G, h, t_max):
    "Using a numerical method called Runge-Kutta of Order 4, calculate the motion of bodies in a N-body system under the influence of gravity"

    # Checks whether the parameters entered are coherent #
    if not all(isinstance(var, (np.ndarray, list)) for var in (masses_list, initial_pos_list, initial_vel_list)):
        raise TypeError("All parameters (masses_list, initial_pos_list, and initial_vel_list) must be of type list or numpy.ndarray.")
    if not all(isinstance(var, (int, float)) for var in (G, h, t_max)):
        raise TypeError("All parameters (G, h, and t_max) must be of type int or float.")
    if not all(var > 0 for var in (G, h, t_max)):
        raise ValueError("All parameters (G, h, and t_max) must be positive values.")
    if not len(masses_list) == len(initial_pos_list) == len(initial_vel_list):
        raise ValueError("The number of elements in masses_list, initial_pos_list, and initial_vel_list must be the same!")
    if np.array(initial_pos_list).shape != (len(masses_list), 3) or np.array(initial_vel_list).shape != (len(masses_list), 3):
        raise ValueError("The dimension of all position and velocity vectors must be (n, 3) where n is the number of bodies.")

    def n_body_acceleration(m_list, current_pos_list):
        "Using masse and current position of all bodies, returns the acceleration suffered by each one"
        bodies_index_list = [i + 1 for i in range(num_bodies)]
        combinations = list(itertools.combinations(bodies_index_list, 2))
        F_dic = {}
        a_list = np.zeros((num_bodies,3), dtype='float128')

        for current_combination in combinations:
            first_i, second_i = current_combination
            pos = current_pos_list[second_i - 1] - current_pos_list[first_i - 1]
            norm = np.linalg.norm(pos)
            F = np.array(masses_list[first_i - 1] * masses_list[second_i - 1] *  pos / (np.array(norm)**3), dtype='float128')
            F_dic[f'{first_i}' + f'{second_i}'] = F

        for body_index in bodies_index_list:
            for combination in F_dic:
                if f'{body_index}' in combination[0]:
                    a_list[body_index-1] += F_dic[combination]
                if f'{body_index}' in combination[1]:
                    a_list[body_index-1] -= F_dic[combination]
            a_list[body_index-1] /= masses_list[body_index-1]
        return G * a_list

    num_steps = int(t_max / h)
    num_bodies = len(masses_list)
    bodies_pos_list = np.zeros((num_steps, num_bodies, 3), dtype='float128')
    bodies_vel_list = np.zeros((num_steps, num_bodies, 3), dtype='float128')
    bodies_pos_list[0] = np.array(initial_pos_list, dtype='float128')
    bodies_vel_list[0] = np.array(initial_vel_list, dtype='float128')

    # RK4 integration method #
    for i in range(1, num_steps):
        accelerations = n_body_acceleration(masses_list, bodies_pos_list[i-1])

        k1_1 = bodies_vel_list[i-1]
        k1_2 = accelerations

        k2_1 = np.array([bodies_vel_list[i-1][j] + 0.5 * h * k1_2[j] for j in range(num_bodies)], dtype='float128')
        k2_2 = n_body_acceleration(masses_list, bodies_pos_list[i-1] + np.array([(h/2) * k for k in k1_1], dtype='float128'))

        k3_1 = np.array([bodies_vel_list[i-1][j] + 0.5 * h * k2_2[j] for j in range(num_bodies)], dtype='float128')
        k3_2 = n_body_acceleration(masses_list, bodies_pos_list[i-1] + np.array([(h/2) * k for k in k2_1], dtype='float128'))

        k4_1 = np.array([bodies_vel_list[i-1][j] + h * k3_2[j] for j in range(num_bodies)], dtype='float128')
        k4_2 = n_body_acceleration(masses_list, bodies_pos_list[i-1] + np.array([h * k for k in k3_1], dtype='float128'))

        for j in range(num_bodies):
            bodies_pos_list[i][j] = np.float128(bodies_pos_list[i-1][j] + (h / 6) * (k1_1[j] + 2 * k2_1[j] + 2 * k3_1[j] + k4_1[j]))
            bodies_vel_list[i][j] = np.float128(bodies_vel_list[i-1][j] + (h / 6) * (k1_2[j] + 2 * k2_2[j] + 2 * k3_2[j] + k4_2[j]))
    return bodies_pos_list, num_steps, num_bodies


def animation_3d(bodies_pos_list, num_frames, num_bodies, plot_scale, title=None, dist=7, marker_sizes=4, colors_list=None, camera_angles=[-40,1], angular_rotation_speeds=[0,0], bodies_names=None, bodies_volume=[None], body_centered_index=None, lw=0.8, font_size=8, animation_speed=50):
    "Makes an animation video of bodies motion"

    # Checks whether the parameters entered are coherent #
    if not all(isinstance(var, int) for var in (num_frames, num_bodies)):
        raise TypeError("num_frames and num_bodies must be integers.")
    if not all(isinstance(var, (int, float)) for var in (lw, font_size, animation_speed)):
        raise TypeError("plot_distance, lw, font_size, and animation_speed must be integers or floats.")
    if not all(isinstance(var, (list, np.ndarray)) for var in (bodies_pos_list, plot_scale, camera_angles, angular_rotation_speeds, bodies_volume)):
        raise TypeError("bodies_pos_list, plot_scale, camera_angles, angular_rotation_speeds, and bodies_volume must be lists or numpy.ndarrays.")
    if not all(isinstance(var, (type(None), list, np.ndarray)) for var in (colors_list, bodies_names, bodies_volume)):
        raise TypeError("colors_list, bodies_names, and bodies_volume must be lists, numpy.ndarrays, or None.")
    if type(body_centered_index) not in [type(None), int, float]:
        raise TypeError("body_centered_index must be None, an integer, or a float.")
    if np.array(bodies_pos_list).shape[0] != num_frames:
        raise ValueError("The number of frames in bodies_pos_list must be equal to num_frames.")
    if np.array(bodies_pos_list).shape[1] != num_bodies:
        raise ValueError("The number of bodies in bodies_pos_list must be equal to num_bodies.")
    if np.array(bodies_pos_list).shape[2] != 3:
        raise ValueError("The dimension of bodies_pos_list must be (num_frames, num_bodies, 3).")
    if np.array(plot_scale).shape != (3,2):
        raise ValueError("The shape of plot_scale must be (3, 2).")
    if np.array(angular_rotation_speeds).shape != (2,):
        raise ValueError("The shape of angular_rotation_speeds must be (2,).")

    # Plot configurations #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if title:
        fig.suptitle(title)
    fig.set_facecolor('gray')
    ax.set_facecolor('gray')
    ax.set_xlim(plot_scale[0])
    ax.set_ylim(plot_scale[1])
    ax.set_zlim(plot_scale[2])
    ax.azim = camera_angles[0]
    ax.elev = camera_angles[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        ax.dist = dist

    lines, scatters, labels = [], [], []
    max_scale = max([max(i) for i in plot_scale])
    if bodies_names == None: bodies_names = [f"Body {i+1}" for i in range(num_bodies)]
    if colors_list == None: colors_list = [None for i in range(num_bodies)]
    for i in range(num_bodies):
        line, = ax.plot(np.array([], dtype='float128'), np.array([], dtype='float128'), np.array([], dtype='float128'), lw=lw, zorder=1, color=colors_list[i])
        lines.append(line)
        scatter, = ax.plot(np.array([], dtype='float128'), np.array([], dtype='float128'), np.array([], dtype='float128'), 'o')
        scatters.append(scatter)
        label = ax.text(0, 0, 0, '', fontsize=font_size)
        labels.append(label)

    # Sub functions #
    def camera_position(angles, scale, center_of_view=[0,0,0]):
        positions = np.array([
        center_of_view[0] + scale * np.sin(np.radians(angles[1])) * np.cos(np.radians(angles[0])),
        center_of_view[1] + scale * np.sin(np.radians(angles[1])) * np.sin(np.radians(angles[0])),
        center_of_view[2] + scale * np.cos(np.radians(angles[1]))])
        return positions

    def size_of_view(volume, angles, scale, center_of_view=[0,0,0], body_pos=np.array([0,0,0]), fov_degrees=60.0):
        distance = np.linalg.norm(camera_position(angles, scale, center_of_view) - body_pos)
        fov_radians = math.radians(fov_degrees)
        distance_order = int(math.floor(math.log10(abs(distance))))
        apparent_size = int(2 * math.tan(0.5 * fov_radians) * (1/abs(distance_order)) * (math.pow(volume, 1/14)))
        if distance_order == 0:   distance_order = 1
        if apparent_size == 0:    return 1
        return apparent_size

    # Animation functions #
    def init():
        for line in lines:
            line.set_data(np.array([], dtype='float128'), np.array([], dtype='float128'))
            line.set_3d_properties(np.array([], dtype='float128'))
        for scatter in scatters:
            scatter.set_data(np.array([], dtype='float128'), np.array([], dtype='float128'))
            scatter.set_3d_properties(np.array([], dtype='float128'))
        for label in labels:
            ax.add_artist(label)
        return lines + scatters + labels

    def update(frame):
        # Set the center on a specific body
        if body_centered_index != None:
            center_x, center_y, center_z = bodies_pos_list[frame, body_centered_index]
            ax.set_xlim(center_x + plot_scale[0][0], center_x + plot_scale[0][1])
            ax.set_ylim(center_y + plot_scale[1][0], center_y + plot_scale[1][1])
            ax.set_zlim(center_z + plot_scale[2][0], center_z + plot_scale[2][1])

        # Update bodies
        for i in range(num_bodies):
            lines[i].set_data(bodies_pos_list[:frame, i, 0], bodies_pos_list[:frame, i, 1])
            lines[i].set_3d_properties(bodies_pos_list[:frame, i, 2])
            line_color = lines[i].get_color()

            x = np.array([bodies_pos_list[frame, i, 0]], dtype='float64')
            y = np.array([bodies_pos_list[frame, i, 1]], dtype='float64')
            z = np.array([bodies_pos_list[frame, i, 2]], dtype='float64')

            scatter = scatters[i]
            scatter.set_data(x, y)
            scatter.set_3d_properties(z)
            scatter.set_color(line_color)

            label = labels[i]
            label.set_text(bodies_names[i])
            label.set_position((x[0], y[0]))
            label.set_rotation(90)
            label.set_z(z[0] + 1e-1)

            # Ajust marker sizes
            if type(marker_sizes) == int:
                scatter.set_markersize(marker_sizes)
            if type(marker_sizes) == list:
                scatter.set_markersize(marker_sizes[i])
            if marker_sizes == 'auto' and not all(bodies_volume):
                raise ValueError('To define the marker_sizes parameter as "auto", you also must pass a list with the volume of the bodies.')
            if marker_sizes == 'auto' and all(bodies_volume):
                scatter.set_markersize(size_of_view(bodies_volume[i], camera_angles, max_scale, [center_x, center_y, center_z], bodies_pos_list[frame, i]))

        # Set rotation effect in camera
        camera_angles[0] += angular_rotation_speeds[0]
        camera_angles[1] += angular_rotation_speeds[1]
        ax.azim = camera_angles[0]
        ax.elev = camera_angles[1]

        # Set the correct display order of scatters and labels (displays bodies in order, based on point of view)
        camera_pos = camera_position(camera_angles, max_scale, [center_x, center_y, center_z])
        view_order = [-1 * np.dot(bodies_pos_list[frame][body], camera_pos) / np.dot(camera_pos,camera_pos) for body in range(num_bodies)]
        sort_list = np.argsort([order for order in view_order])
        sort_dic = {index: i for i, index in enumerate(sort_list)}
        correct_sorted_list = [sort_dic[i] for i in range(num_bodies)]
        for i, scatter, label in zip(correct_sorted_list, scatters, labels):
            scatter.set_zorder(i + 2)  # Use i+2 to ensure that are above the lines
            label.set_zorder(i + 2)

        return lines + scatters + labels

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=animation_speed)
    return HTML(ani.to_html5_video())
