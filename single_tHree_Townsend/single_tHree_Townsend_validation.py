#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

"""Plot particles at a given frame number for an NPZ file specified in the
script.

Does not plot any periodic copies. If you want to do this, see the code in
plot_particle_positions_video.py.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pylab import rcParams
sys.path.append("..")  # Allows importing from SD directory
#from functions.graphics import (plot_all_spheres, plot_all_dumbbells,
#                                plot_all_torque_lines, plot_all_velocity_lines,
 #                               plot_all_angular_velocity_lines)
#from functions.shared import add_sphere_rotations_to_positions
import matplotlib.colors as colors
import matplotlib        as mpl
import matplotlib.cm     as cm

plt.style.use('seaborn-v0_8-colorblind');


plt.rc('text',usetex=True);
plt.rc('font',family='serif');

cmap = cm.get_cmap('plasma');

DATA = np.genfromtxt('single_tHree_Townsend.txt',dtype='str')
filename = [sub[: -4] for sub in DATA]

e=1
frameno = 0
dt= 0.5
num_frame=2
V=[]
T=[]
viewing_angle = (0, -90)
viewbox_bottomleft_topright = np.array([[-15, -15, -15], [15, 15, 15]])
two_d_plot = False
view_labels = False
trace_paths = 0
for i in range(len(filename)):
    v=[]
    t=[]
    # Naming the folders like this means you can run this script from any directory
    this_folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = this_folder + "/output/"

    data1 = np.load(output_folder + filename[i] + ".npz")
    positions_centres = data1['centres']
    positions_deltax = data1['deltax']
    Fa_out = data1['Fa']
    Fb_out = data1['Fb']
    DFb_out = data1['DFb']

    num_frames = positions_centres.shape[0]
    num_particles = positions_centres.shape[1]
    num_dumbbells = positions_deltax.shape[1]
    num_spheres = num_particles - num_dumbbells
    
    for frameno in range(0, num_frame):
        sphere_positions = positions_centres[frameno, 0:num_spheres, :]
        dumbbell_positions = positions_centres[frameno, num_spheres:num_particles, :]
        dumbbell_deltax = positions_deltax[frameno, :, :]

        sphere_sizes = np.array([1 for _ in range(num_spheres)])
        dumbbell_sizes = np.array([0.1 for _ in range(num_dumbbells)])
        sphere_rotations=0
#        sphere_rotations = add_sphere_rotations_to_positions(
#           sphere_positions, sphere_sizes, np.array([[1, 0, 0], [0, 0, 1]]))
        Ta_out = [[0, 0, 0] for _ in range(num_spheres)]
        Oa_out = [[0, 0, 0] for _ in range(num_spheres)]
        Ua_out = [[0, 0, 0] for _ in range(num_spheres)]

        posdata = [sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
               dumbbell_positions, dumbbell_deltax]
        if frameno != 0:
            t.append(dt*frameno)
            vxyz=np.sqrt(np.sum((posdata[1][e]-previous_step_posdata[1][e])**2))/dt
            v.append(vxyz)
        previous_step_posdata = posdata
    V.append(v)
    T.append(t)
    e=1
# Pictures initialise
rcParams.update({'font.size': 11})
rcParams.update({'figure.dpi': 120, 'figure.figsize': [6, 6],
                 'savefig.dpi': 140})
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.view_init(viewing_angle[0], viewing_angle[1])
spheres = list()
dumbbell_lines = list()
dumbbell_spheres = list()
force_lines = list()
force_text = list()
torque_lines = list()
velocity_lines = list()
velocity_text = list()
sphere_labels = list()
angular_velocity_lines = list()
sphere_lines = list()
sphere_trace_lines = list()
dumbbell_trace_lines = list()
#v = viewbox_bottomleft_topright.transpose()
#ax.auto_scale_xyz(v[0], v[1], v[2])
#ax.set_xlim(v[0, 0], v[0, 1])
#ax.set_ylim(v[1, 0], v[1, 1])
#ax.set_zlim(v[2, 0], v[2, 1])
#ax.set_box_aspect((1, 1, 1), zoom=1.4)
if two_d_plot:
    ax.set_proj_type('ortho')
    ax.set_yticks([])
else:
    ax.set_ylabel("$V$")
ax.set_xlabel("$d/R$")
#ax.set_zlabel("$z$")
fig.tight_layout()

# Pictures
beta=1.3
al=np.arctan(beta)
Mf1 = 100/(6*1*np.pi*1*100)
Mf2 = Mf1/1.10
V=np.array(V)
V=V/Mf1
R=np.logspace(-1,3, num=300)
R=np.array(R)

An_0=(1-(3*(np.sin(al))**2)/(2*(R)) - (3*(np.cos(al))**2)/(4*(R)) +((np.sin(al))**2)/(2*((R))**3) -((np.cos(al))**2)/(4*((R))**3))
V_1_1=[V[i][0] for i in range(0,300)]
V_1_10=[V[i+300][0] for i in range(0,300)]
V_1_100=[V[i+600][0] for i in range(0,300)]


plt.plot(R, V_1_1, marker = 'v', linestyle=' ', color=cmap(0.1), label = 'radius 1/1 ')
plt.plot(R, V_1_10, marker = '*', linestyle=' ', color=cmap(0.2), label = 'radius 1/10 ')
plt.plot(R, V_1_100, marker = '.', linestyle=' ', color=cmap(0.3), label = 'radius 1/100' )
plt.plot(R, An_0, marker = ' ', linestyle='-', color=cmap(0.25), label = 'Analytical point force')

#plt.plot(Rf, Vfn, marker = 's', linestyle= ' ', color= cmap(0.001), label= 'Froce-free Picella $\\beta$=0')
ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.legend()
ax.set_title('Speed of Microswimmers as a function the distance \n between the main sphere and the spheres nearby', loc='center', y=1.055, fontsize=11)
plt.tight_layout()
np.savez('Townsend_velocity_'+str(int(time.time())), V_1_1, V_1_10, V_1_100)
plt.savefig('output/figure'+str(int(time.time())) +'Comparaison distance.pdf', format='pdf', dpi=300 )
plt.show()
