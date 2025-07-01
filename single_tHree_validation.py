import pystokes
import numpy as np

""" tHree definition for pystokes
    for the moment, particles are all aligned in the y direction only."""


""" For the moment, simple one, all pointing upwards...  """
def tHree_build_full_positions_array(NtHree, rBody, d, alpha):
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


def tHree_build_full_forces_array(NtHree, T, B):

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


def single_tHree_analytic(F, mu, a, d, alpha):
    sin2 = np.sin(alpha)**2
    cos2 = np.cos(alpha)**2

    prefactor = F / (6 * np.pi * mu * a)

    correction = (
        1
        - (3 * a * sin2) / (2 * d)
        - (3 * a * cos2) / (4 * d)
        + (a**3 * sin2) / (2 * d**3)
        - (a**3 * cos2) / (4 * d**3)
    )

    V = prefactor * correction
    return V



NtHree = 1 # number of tHree microswimmers
a = 1      # radius of the particles composing a tHree system
eta = 1    # viscosity
alpha = np.pi / 4 ### $\alpha>0$ puller, $\alpha=0$ neutra, $\alpha<0$ pusher$

# Initial location of the tHree bodies (not the trans-cis flagella...)

rBody = np.array([0,     # x of body 0 and 1
                  0,     # y of body 0 and 1
                  0])    # z of body 0 and 1

### Constant quantities for the present test.
F = tHree_build_full_forces_array(NtHree,1,0);
# The unknown now is the velocity array for the bodies only...
# but first, I have to compute it for the full system.
v = np.zeros(NtHree*9); # each tHree microswimmer is represented by 3 particles
                          # each one having 3 deegrees of freedom;
                          # hence, the *9 factor
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta) # initialize class
                                                                             # only once!



""" Loop over an _arbitrary_ distance """
d = 0*a+np.logspace(-1, +3, num=300)  # from [0.01,100] +2.*a, to avoid contact
# prepare empty list to store computed values
ux = []; # velocity, in x direction
uy = []; # velocity, in y direction

for d_i in d:
    # full position will depend on the body--cis-trans-flagella distance
    r = tHree_build_full_positions_array(NtHree, rBody, d_i, alpha)
    # Provided the geometrical configuration (r, full positions array)
    # and the forces applied to the system (F, full forces array)
    # determine the velocities of each particle. Translation velocities only.
    # but first, set velocity to zero.
    v = v*0.;
    # only now I can compute everything using Stokesian Dynamics
    rbm.mobilityTT(v,r,F)
    # Recover the velocity array of the bodies only.
    vBody = extract_arrayBody(v,NtHree); 
    #print(vBody);
    ux.append(vBody[0]*6.*np.pi)
    uy.append(vBody[1]*6.*np.pi) # normalized wrt the sedimentation speed of a particle

# Now provide a simple plot...
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
plt.rc('text',usetex=True);
plt.rc('font',family='serif');

fig, ax1 = plt.subplots(1,1, figsize=(4,3));

cmap = cm.get_cmap('plasma');

ax1.plot(d/a,uy,'-',color=cmap(0.0),label='PyStokes');

# get the values obtained using the analytical prediction...
uyANA = single_tHree_analytic(1, eta, a, d, alpha)
ax1.plot(d/a,uyANA*6*np.pi,'--',color=cmap(0.75),label='analytic');

# add shaded area, to indicate that values below a given threshold are not physical.
# Add shaded background for x < 2
plt.axvspan(xmin=d[0]/a, xmax=2, color='gray', alpha=0.3)



ax1.legend();
ax1.set_xlabel('$d/R$')
ax1.set_ylabel('$U_{swim}$')
# Set logarithmic scale
plt.xscale('log')   # Log scale on X-axis
plt.yscale('log')   # Log scale on Y-axis
plt.tight_layout();
plt.savefig('single_tHree_validation_pystokes.png',format='svg');

