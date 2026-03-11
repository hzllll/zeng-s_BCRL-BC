import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight


    def collision_check(self, paths, obstacles):

        collision_check_array = np.zeros(len(paths), dtype=bool)
        if obstacles == []:
            return collision_check_array
        for i in range(len(paths)):
            collision_free = True
            path           = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free = collision_free and \
                                     not np.any(collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break

            collision_check_array[i] = collision_free

        return collision_check_array

    def select_best_path_index(self, paths, collision_check_array, goal_state):

        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:

                path = paths[i]
                x_diff = path[0][-1] - goal_state[0]
                y_diff = path[1][-1] - goal_state[1]
                score = np.sqrt(x_diff**2 + y_diff**2)

                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:

                            pass

            # Handle the case of colliding paths.
            else:
                score = float('Inf')

            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index
