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
# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)
d = 3.3*a     # distance between the tHree body and flagella (trans and cis one, represented by a particle)
beta = 1.3
alpha = np.arctan(beta) ### $\alpha>0$ puller, $\alpha=0$ neutra, $\alpha<0$ pusher$

Thrust   = 1. # thrust of the tHree microswimmer
Buoyancy = -Thrust*np.array([0.,0.2,0.5,0.6]) # buoyancy applied on the body of the microswimmer 

# Initialize class
rbm    = pystokes.unbounded.Rbm(radius=a, particles=NtHree*3, viscosity=eta)

# prepare empty list to store computed values
ux = [[] for i in range(len(Buoyancy))]; # velocity, in x direction
uy = [[] for i in range(len(Buoyancy))]; # velocity, in y direction
uz = [[] for i in range(len(Buoyancy))];
# loop over an arbitrary distance 
LAMBDA = 2*d*np.cos(alpha)*0.5 + np.logspace(-1,+2,num=100)

for i in range(len(Buoyancy)):
      for lambda_i in LAMBDA:
            # Initialize forces applied on the system...
            F = tHree.full_forces(NtHree,Thrust,Buoyancy[i]);
            rBody = np.array([0, lambda_i,     # x of body 0 and 1
                      0, 0       ,     # y of body 0 and 1
                      0, 0       ])    # z of body 0 and 1
            r = tHree.full_positions(NtHree, rBody, d, alpha)
            v = np.zeros(NtHree*9) 
            rbm.mobilityTT(v,r,F)
            vBody = tHree.extract_arrayBody(v,NtHree)
            ux[i].append(vBody[0]);
            uy[i].append(vBody[2]);
            uz[i].append(vBody[4]);
#    print(vBody);

# Now provide a simple plot...
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
plt.rc('text',usetex=True);
plt.rc('font',family='serif');

fig, ax1 = plt.subplots(1,1, figsize=(4,3));

cmap = cm.get_cmap('plasma');

#Townsend plot

npzfile=np.load('buoyancy_binary_interactions_Townsend/Townsend_buoyancy_binary_velocity_20250717155955.npz')
V_y_mono_0=npzfile['arr_0']
V_y_0=npzfile['arr_1']
V_y_mono_2=npzfile['arr_2']
V_y_2=npzfile['arr_3']
V_y_mono_5=npzfile['arr_4']
V_y_5=npzfile['arr_5']
V_y_mono_6=npzfile['arr_6']
V_y_6=npzfile['arr_7']
Lambda=npzfile['arr_8']
ax1.plot(Lambda, V_y_mono_0/np.abs(V_y_mono_0[-1]), '.', markersize=2, color=cmap(0.90), label='Townsend $U_{swim}$ 1/1, B=0F')
ax1.plot(Lambda, V_y_0/np.abs(V_y_0[-1]), '*',markersize=2,  color=cmap(0.80), label='Townsend $U_{swim}$ 1/10, B=0F')
ax1.plot(Lambda, V_y_mono_2/np.abs(V_y_mono_2[-1]),  '.', markersize=2, color=cmap(0.92), label='Townsend $U_{swim}$ 1/1, B=.2F')
ax1.plot(Lambda, V_y_2/np.abs(V_y_2[-1]), '*', markersize=2, color=cmap(0.82), label='Townsend $U_{swim}$ 1/10, B=.2F')
ax1.plot(Lambda, V_y_mono_5/np.abs(V_y_mono_5[-1]),  '.', markersize= 2, color=cmap(0.95), label='Townsend $U_{swim}$ 1/1, B=.5F')
ax1.plot(Lambda, V_y_5/np.abs(V_y_5[-1]), '*',markersize=2,  color=cmap(0.85), label='Townsend $U_{swim}$ 1/10, B=.5F')
ax1.plot(Lambda, V_y_mono_6/np.abs(V_y_mono_6[-1]), '.',markersize=2, color=cmap(0.96), label='Townsend $U_{swim}$ 1/1, B=.6F')
ax1.plot(Lambda, V_y_6/np.abs(V_y_6[-1]), '*',markersize=2, color=cmap(0.86), label='Townsend $U_{swim}$ 1/10, B=.6F')


#Analytic
def binary_interaction_analytic(F, B, mu, a, d, alpha, lam):
    prefactor=1/(6*np.pi*mu*a)
    
    #single swimmer

    U_swim_flagel_1, U_side_flagel_1=tHree.point_force_Analytic(d,alpha,-F/2,0,a)
    U_swim_flagel_2, U_side_flagel_2=tHree.point_force_Analytic(d,np.pi-alpha,-F/2,0,a)
    An_single_swim = F+B+U_swim_flagel_1 +U_swim_flagel_2
    An_single_side = U_side_flagel_1 +U_side_flagel_2
    #secondary swimmer
    
    U_swim_body, U_side_body = tHree.point_force_Analytic(lam,0,F+B,0,a)
    #first flagel

    l_flagel_1=np.sqrt(d**2+lam**2-2*lam*d*np.cos(alpha))
    beta_flagel_1=np.arcsin(d*np.sin(alpha)/l_flagel_1)
    U_swim_flagel_1, U_side_flagel_1=tHree.point_force_Analytic(l_flagel_1,beta_flagel_1,-F/2,0,a)

    #second flagel

    l_flagel_2=np.sqrt(d**2+lam**2+2*lam*d*np.cos(alpha))
    beta_flagel_2=np.arcsin(d*np.sin(alpha)/l_flagel_2)
    U_swim_flagel_2, U_side_flagel_2=tHree.point_force_Analytic(l_flagel_2,beta_flagel_2,-F/2,0,a)

    An_binary_swim= An_single_swim + U_swim_body+U_swim_flagel_1+U_swim_flagel_2
    An_binary_side= An_single_side + U_side_body+U_side_flagel_1+U_side_flagel_2
   
    V_swim=prefactor*An_binary_swim
    V_side=prefactor*An_binary_side

    return V_swim, V_side


for i in range(len(Buoyancy)):
    uyANA,uxANA=binary_interaction_analytic(Thrust,Buoyancy[i], eta, a, d, alpha, LAMBDA)
    ax1.plot(LAMBDA/a,uy[i]/np.abs(uy[i][-1]),'-',color=cmap(0.59+i/25),label='$U_{swim}$, B='+str(-Buoyancy[i]/Thrust)+'F');
    ax1.plot(LAMBDA/a,uyANA/np.abs(uyANA[-1]),'--',color=cmap(0.1+i/25),label='analytic $U_{swim}$, B='+str(-Buoyancy[i]/Thrust) +'F');




### Gray-out the area for lambda associated to particle inter-penetration!
plt.axvspan(xmin=LAMBDA[0]/a,xmax=2*d*np.cos(alpha), color='gray',alpha=0.3);
plt.hlines(0,LAMBDA[0]/a,LAMBDA[-1]/a, color=cmap(0.99))
ax1.legend(fontsize='xx-small');
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
plt.savefig('buoyancy_binary_interactions'+str(timestamp)+'.svg',format='svg');
