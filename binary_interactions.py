import pystokes
import numpy as np
import tHree

NtHree = 2 # number of tHree microswimmers
a = 1      # radius of the particles composing a tHree system
eta = 1    # viscosity

# The unknown now is the velocity array for the bodies only...
# but first, I have to compute it for the full system.
v = np.zeros(NtHree*9); # each tHree microswimmer is represented by 3 particles
                          # each one having 3 deegrees of freedom;
                          # hence, the *9 factor
# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)
d = 2.*a     # distance between the tHree body and flagella (trans and cis one, represented by a particle)
alpha = np.pi / 4 ### $\alpha>0$ puller, $\alpha=0$ neutra, $\alpha<0$ pusher$

Thrust   = 1. # thrust of the tHree microswimmer
Buoyancy = -0. # buoyancy applied on the body of the microswimmer 

# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)
# Initialize forces applied on the system...
F = tHree.full_forces(NtHree,Thrust,Buoyancy);

# prepare empty list to store computed values
ux = []; # velocity, in x direction
uy = []; # velocity, in y direction
uz = [];
# loop over an arbitrary distance 
LAMBDA = 2*d*np.cos(alpha)*0.5 + np.logspace(-1,+2,num=100)
for lambda_i in LAMBDA:

    rBody = np.array([0, lambda_i,     # x of body 0 and 1
                      0, 0       ,     # y of body 0 and 1
                      0, 0       ])    # z of body 0 and 1
    r = tHree.full_positions(NtHree, rBody, d, alpha)
    v = np.zeros(NtHree*9) 
    rbm.mobilityTT(v,r,F)
    vBody = tHree.extract_arrayBody(v,NtHree)
    ux.append(vBody[0]);
    uy.append(vBody[2]);
    uz.append(vBody[4]);
#    print(vBody);

# Now provide a simple plot...
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
plt.rc('text',usetex=True);
plt.rc('font',family='serif');

fig, ax1 = plt.subplots(1,1, figsize=(4,3));

cmap = cm.get_cmap('plasma');

ax1.plot(LAMBDA/a,uy/np.abs(uy[-1]),'-',color=cmap(0.4),label='$U_{swim}$');
ax1.plot(LAMBDA/a,ux/np.abs(uy[-1]),'-',color=cmap(0.9),label='$U_{side}$');

### Gray-out the area for lambda associated to particle inter-penetration!
plt.axvspan(xmin=LAMBDA[0]/a,xmax=2*d*np.cos(alpha), color='gray',alpha=0.3);

ax1.legend();
ax1.set_xlabel('$\lambda/R$')
# Set logarithmic scale
plt.xscale('log')   # Log scale on X-axis
#plt.yscale('log')   # Log scale on Y-axis

lambda_contact = 2.*np.cos(alpha)*d;

# Get current xticks and labels
xticks = list(plt.xticks()[0])
xticklabels = [item.get_text() for item in plt.gca().get_xticklabels()]

# Add the new tick
xticks.append(lambda_contact)
xticklabels.append(r'$\frac{\tilde{\lambda}}{R}$')  # custom LaTeX label

# Set new ticks and labels
plt.xticks(xticks, xticklabels)
plt.xlim(np.min(LAMBDA/a),np.max(LAMBDA/a));

plt.tight_layout();
plt.savefig('binary_interactions.svg',format='svg');
