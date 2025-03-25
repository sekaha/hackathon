import numpy as np

def get_random_spherical(r):
    # to uniformly sample for phi you do need to use acos, otherwise it bunches around the poles
    return spherical_to_cartesian(
        r, np.random.uniform(0, 2 * np.pi), np.acos(np.random.uniform(-1, 1))
    )


# https://en.wikipedia.org/wiki/Spherical_coordinate_system
def spherical_to_cartesian(r, theta, phi):
    coords = np.zeros(3)
    coords[0] = np.cos(theta) * np.sin(phi)
    coords[1] = np.sin(theta) * np.sin(phi)
    coords[2] = np.cos(phi)

    return coords * r
