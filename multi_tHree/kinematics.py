import pystokes
import numpy as np
import tHree
from tqdm import tqdm # for a nice completion bar ;)

### DOMAIN FEATURES
L = 64 # domain size
Nb,Nm = 1, 4 # parameters associated to the periodic approach.

### FLUID FEATURES
a = 1.      # radius of the particles composing a tHree system
eta = 1./(np.pi*6)    # viscosity, set so that 
                      # sedimentation velocity is 1 with force set to 1

### tHree FEATURES
NtHree = 10 # number of tHree microswimmers
d = 1.*a     # distance between the tHree body and flagella 
alpha = np.pi / 4 ### $\alpha>0$ puller, $\alpha=0$ neutra, $\alpha<0$ pusher$

Thrust   = 1. # thrust of the tHree microswimmer
Buoyancy = -0. # buoyancy applied on the body of the microswimmer

### TIMESTEPPING
dt = 2.5
t_final = 100

# Initialize PyStokes class
rbm = pystokes.periodic.Rbm(a,NtHree*3,eta,L)
# Initialize forces and velocities applied on the system...
F = tHree.full_forces(NtHree,Thrust,Buoyancy)
v = np.zeros(NtHree*9); 
# Initialize particle location, start with body centers.
rBody = tHree.random_body_positions(NtHree,L) # domain is defined from 0 to L...
vBody = tHree.extract_arrayBody(v,NtHree)

print(rBody)

### PREPARING TIMESTEPPING
n_steps = int(t_final / dt) + 1
t = np.linspace(0, t_final, n_steps)
### Allocate variables containing the results
# 6 components: x, y, z, vx, vy, vz
kinematics = np.zeros((n_steps, NtHree, 6))

### assign initial condition.
kinematics[0] = np.vstack((rBody, vBody)).reshape(6, NtHree).T

### TIMESTEPPING
for step in tqdm(range(1, n_steps), desc="Time stepping"):
    # Recover rBody from previous step
    rBody, vBody = np.split(kinematics[step-1].T.reshape(-1), 2)
    # Reconstruct the full position array
    r = tHree.full_positions(NtHree, rBody, d, alpha)
    # Reconstruct full velocity array
    v = np.zeros(NtHree*9)
    # APPLY RBM from Stokesian Dynamics
    rbm.mobilityTT(v,r,F)
    # Get only the velocities associated to the bodies
    vBody = tHree.extract_arrayBody(v,NtHree)
    # Advance the locations of the bodies only, using Explicit Euler
    rBody+=vBody*dt;
    # Impose periodicity constraint, on all directions
    rBody %= L ### super compact, wow
    # write back the updated body locations in the kinematics array
    kinematics[step] = np.vstack((rBody, vBody)).reshape(6, NtHree).T

### EXPORT KINEMATICS
tHree.export_kinematics_per_particle(kinematics,t)
