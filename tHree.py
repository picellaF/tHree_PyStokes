#import pystokes
import numpy as np

print("Loading tHree modelling routines.")
print("francesco.picella@sorbonne-universite.fr")
print("summer 2025")

""" tHree definition for pystokes
    for the moment, particles are all aligned in the y direction only."""

### for an usage example, please refer to "single_tHree_validation.py"

""" For the moment, simple one, all pointing upwards...  """
def full_positions(NtHree, rBody, d, alpha):
    N = 3 * NtHree  # total number of particles
    dim = 3

    # Extract x, y, z coordinates of body particles
    x_body = rBody[0:NtHree]
    y_body = rBody[NtHree:2*NtHree]
    z_body = rBody[2*NtHree:3*NtHree]

    # Preallocate arrays for full positions
    x_full = np.zeros(N)
    y_full = np.zeros(N)
    z_full = np.zeros(N)

    # Fill in body positions
    for i in range(NtHree):
        base = 3 * i
        x_full[base] = x_body[i]
        y_full[base] = y_body[i]
        z_full[base] = z_body[i]

        # Trans position
        x_full[base + 1] = x_body[i] + d * np.cos(alpha)
        y_full[base + 1] = y_body[i] + d * np.sin(alpha)
        z_full[base + 1] = z_body[i]

        # Cis position
        x_full[base + 2] = x_body[i] - d * np.cos(alpha)
        y_full[base + 2] = y_body[i] + d * np.sin(alpha)
        z_full[base + 2] = z_body[i]

    # Concatenate into one array: [x0, x1,... xN-1, y0, ..., yN-1, z0, ..., zN-1]
    rFull = np.concatenate([x_full, y_full, z_full])
    return rFull


def full_forces(NtHree, T, B):

    """ 
    T is the _thrust_ developed by the tHree microswimmer,
    while B is the buoyancy force, applied to the center body only."""

    N = 3 * NtHree
    dim = 3

    # Initialize force arrays
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)

    for i in range(NtHree):
        base = 3 * i

        # Apply forces in y-direction
        Fy[base]     = T + B      # Body
        Fy[base + 1] = -T / 2     # Trans
        Fy[base + 2] = -T / 2     # Cis

    # Concatenate to form full F vector
    F = np.concatenate([Fx, Fy, Fz])
    return F

def extract_arrayBody(v, NtHree):
    N = 3 * NtHree
    dim = 3

    # Indices of body particles: 0, 3, 6, ..., 3*(NtHree-1)
    body_indices = np.arange(0, N, 3)

    # Split velocity components
    vx = v[0:N]
    vy = v[N:2*N]
    vz = v[2*N:3*N]

    # Extract components for bodies only
    vx_body = vx[body_indices]
    vy_body = vy[body_indices]
    vz_body = vz[body_indices]

    # Concatenate in the same format: [vx_body, vy_body, vz_body]
    vBody = np.concatenate([vx_body, vy_body, vz_body])
    return vBody
