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

def point_force_Analytic(l,beta,fx,fy,a):
    cos2=(np.cos(beta))**2
    sin2=(np.sin(beta))**2
    cossin=np.cos(beta)*np.sin(beta)
    U_swimx=cos2*(1/4)*(3*a/l + (a/l)**3) +sin2*(1/2)*(3*a/l-(a/l)**3)
    U_swimx=fx*U_swimx
    U_sidex=cossin*(1/4)*(3*a/l-(a/l)**3)
    U_sidex=fx*U_sidex
    U_swimy=cossin*(3/4)*(a/l-(a/l)**3)
    U_swimy=fy*U_swimy
    U_sidey=sin2*(1/4)*(3*a/l +(a/l)**3) +cos2*(1/2)*(3*a/l-(a/l)**3)
    U_sidey=fy*U_sidey
    U_swim = U_swimx +U_swimy
    U_side = U_sidex + U_sidey

    return U_swim, U_side






def full_forces_w_2Drotation(NtHree, T, B, angle):

    """ 
    T is the _thrust_ developed by the tHree microswimmer,
    while B is the buoyancy force, applied to the center body only.
    angle is the angle compared to the vertical array NtHree size"""


    N = 3 * NtHree
    dim = 3

    # Initialize force arrays
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)

    for i in range(NtHree):
        base = 3 * i

        # Apply forces in y-direction
        Fy[base]     = T*np.cos(angle[i]) + B      # Body
        Fx[base]     =- T*np.sin(angle[i])       # Body
        Fy[base + 1] = -T*np.cos(angle[i]) / 2     # Trans
        Fy[base + 2] = -T*np.cos(angle[i]) / 2     # Cis
        Fx[base + 1] = T*np.sin(angle[i]) / 2     # Trans
        Fx[base + 2] = T*np.sin(angle[i]) / 2     # Cis

    # Concatenate to form full F vector
    F = np.concatenate([Fx, Fy, Fz])
    return F



def full_positions_w_2Drotation(NtHree, rBody, d, alpha,angle):
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
        x_full[base + 1] = x_body[i] + d * np.cos(alpha+angle[i])
        y_full[base + 1] = y_body[i] + d * np.sin(alpha+angle[i])
        z_full[base + 1] = z_body[i]

        # Cis position
        x_full[base + 2] = x_body[i] - d * np.cos(alpha-angle[i])
        y_full[base + 2] = y_body[i] + d * np.sin(alpha-angle[i])
        z_full[base + 2] = z_body[i]

    # Concatenate into one array: [x0, x1,... xN-1, y0, ..., yN-1, z0, ..., zN-1]
    rFull = np.concatenate([x_full, y_full, z_full])
    return rFull
