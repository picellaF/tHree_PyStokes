import pystokes
import numpy as np
import tHree
import datetime

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")


NtHree = 2 # number of tHree microswimmers
a = 1      # radius of the particles composing a tHree system
eta = 1    # viscosity

# The unknown now is the velocity array for the bodies only...
# but first, I have to compute it for the full system.
v = np.zeros(NtHree*9); # each tHree microswimmer is represented by 3 particles
                          # each one having 3 deegrees of freedom;
                          # hence, the *9 factor
o=np.zeros(NtHree*9)
# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)
d = 3.3*a     # distance between the tHree body and flagella (trans and cis one, represented by a particle)
beta = 1.3
alpha = np.arctan(beta) ### $\alpha>0$ puller, $\alpha=0$ neutra, $\alpha<0$ pusher$

Thrust   = 1. # thrust of the tHree microswimmer
Buoyancy = -0. # buoyancy applied on the body of the microswimmer 

# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)
#Initialize relative position (2D system)
angle=[0,0]
# Initialize forces applied on the system...
F = tHree.full_forces(NtHree,Thrust,Buoyancy);

# prepare empty list to store computed values
ux = []; # velocity, in x direction
ox = []; # velocity, in x direction
uy = []; # velocity, in y direction
uz = [];
oy = []; # velocity, in y direction
oz = [];
# loop over an arbitrary distance 
LAMBDA = 2*d*np.cos(alpha)*0.5 + np.logspace(-1,+2,num=100)
for lambda_i in LAMBDA:

    rBody = np.array([0, lambda_i,     # x of body 0 and 1
                      0, 0       ,     # y of body 0 and 1
                      0, 0       ])    # z of body 0 and 1
    r = tHree.full_positions(NtHree, rBody, d, alpha)
    v = np.zeros(NtHree*9)
    o = np.zeros(NtHree*9)
    rbm.mobilityTT(v,r,F)
    rbm.mobilityRT(o,r,F)
    oBody = tHree.extract_arrayBody(o,NtHree)
    vBody = tHree.extract_arrayBody(v,NtHree)
    ux.append(vBody[0]);
    ox.append(oBody[0]);
    uy.append(vBody[2]);
    uz.append(vBody[4]);
    oy.append(oBody[2]);
    oz.append(oBody[4]);
    print(oBody);

# Now provide a simple plot...
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
plt.rc('text',usetex=True);
plt.rc('font',family='serif');

fig, ax1 = plt.subplots(1,1, figsize=(4,3));

cmap = cm.get_cmap('plasma');

#Townsend plot

npzfile=np.load('binary_interactions_Townsend/Townsend_binary_velocity_1752145687.npz')
V_y_mono=npzfile['arr_0']
V_y=npzfile['arr_1']
V_x_mono=npzfile['arr_2']
V_x=npzfile['arr_3']
Lambda=npzfile['arr_4']

ax1.plot(Lambda, V_y_mono/np.abs(V_y_mono[-1]), '.', color=cmap(0.99), label='Townsend $U_{swim}$ 1/1')
ax1.plot(Lambda, V_x_mono/np.abs(V_y_mono[-1]), '.', color=cmap(0.99), label='Townsend $U_{side}$ 1/1')
ax1.plot(Lambda, V_y/np.abs(V_y[-1]), '.', color=cmap(0.89), label='Townsend $U_{swim}$ 1/10')
ax1.plot(Lambda, V_x/np.abs(V_y[-1]), '.', color=cmap(0.89), label='Townsend $U_{side}$ 1/10')

#Analytic
def binary_interaction_analytic(F, mu, a, d, alpha, lam):
    prefactor=1/(6*np.pi*mu*a)
    
    Fx=F
    Fy=0
    #single swimmer

    U_swim_flagel_1, U_side_flagel_1=tHree.point_force_Analytic(d,alpha,-Fx/2, -Fy/2,a)
    U_swim_flagel_2, U_side_flagel_2=tHree.point_force_Analytic(d,np.pi-alpha,-Fx/2, -Fy/2,a)
    An_single_swim = Fx+U_swim_flagel_1 +U_swim_flagel_2
    An_single_side = Fy+U_side_flagel_1 +U_side_flagel_2
    #secondary swimmer
    
    U_swim_body, U_side_body = tHree.point_force_Analytic(lam,0,Fx,Fy,a)
    #first flagel

    l_flagel_1=np.sqrt(d**2+lam**2-2*lam*d*np.cos(alpha))
    beta_flagel_1=np.arcsin(d*np.sin(alpha)/l_flagel_1)
    U_swim_flagel_1, U_side_flagel_1=tHree.point_force_Analytic(l_flagel_1,beta_flagel_1,-Fx/2, -Fy/2,a)

    #second flagel

    l_flagel_2=np.sqrt(d**2+lam**2+2*lam*d*np.cos(alpha))
    beta_flagel_2=np.arcsin(d*np.sin(alpha)/l_flagel_2)
    U_swim_flagel_2, U_side_flagel_2=tHree.point_force_Analytic(l_flagel_2,beta_flagel_2,-Fx/2, -Fy/2,a)

    An_binary_swim= An_single_swim + U_swim_body+U_swim_flagel_1+U_swim_flagel_2
    An_binary_side= An_single_side + U_side_body+U_side_flagel_1+U_side_flagel_2
   
    V_swim=prefactor*An_binary_swim
    V_side=prefactor*An_binary_side

    return V_swim, V_side

uyANA,uxANA=binary_interaction_analytic(Thrust, eta, a, d, alpha, LAMBDA)



ax1.plot(LAMBDA/a,uy/np.abs(uy[-1]),'-',color=cmap(0.4),label='$U_{swim}$');
ax1.plot(LAMBDA/a,ux/np.abs(uy[-1]),'-',color=cmap(0.5),label='$U_{side}$');
ax1.plot(LAMBDA/a,oz/np.abs(uy[-1]),'-',color=cmap(0.6), label = '$\\omega$')

ax1.plot(LAMBDA/a,uyANA/np.abs(uyANA[-1]),'--',color=cmap(0.15),label='analytic $U_{swim}$');
ax1.plot(LAMBDA/a,uxANA/np.abs(uyANA[-1]),'--',color=cmap(0.25),label='analytic $U_{side}$');




### Gray-out the area for lambda associated to particle inter-penetration!
plt.axvspan(xmin=LAMBDA[0]/a,xmax=2*d*np.cos(alpha), color='gray',alpha=0.3);

ax1.legend();
ax1.set_xlabel('$\\lambda/R$')
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
plt.savefig('binary_interactions_w_rotation'+str(timestamp)+'.svg',format='svg');
plt.show()
