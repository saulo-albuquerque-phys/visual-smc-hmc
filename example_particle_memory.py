import numpy as np
import matplotlib.pyplot as plt
from SMC_particle_memory.SMC_particle_memory import HMC_one_mode

x1,y1=-5,5
x2,y2=5,-5
alpha_x1,alpha_y1=8,3
alpha_x2,alpha_y2=2,3

PRIOR_ranges_X=(-20,20)
PRIOR_ranges_Y=(-20,20)
likelihood_parameters={'x1':x1,'x2':x2,'y1':y1,'y2':y2,'alpha_x1':alpha_x1,'alpha_x2':alpha_x2,'alpha_y1':alpha_y1, 'alpha_y2':alpha_y2}
ess_rate=0.9
n_particles=1000

hmc=HMC_one_mode(PRIOR_ranges_X,PRIOR_ranges_Y,likelihood_parameters,ess_rate,n_particles)

N_hmc_particles=100
momentum_space_size=0.15

hmc.randomly_generate_particles_from_prior(N_hmc_particles,factor_momentum=momentum_space_size)

particles=hmc.particles

delta_time=0.001
n_steps=10000

x_min,x_max=PRIOR_ranges_X
y_min,y_max=PRIOR_ranges_Y

x = np.linspace(x_min, x_max, 1000)
y = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x, y)

Z = hmc_nm.likelihood(X, Y)

cmap = plt.get_cmap('viridis')
vmin = np.min(Z)
vmax = np.max(Z)

delta_time=0.001
n_steps=2000

N_iterations=20

all_positions,all_likelihoods=hmc_nm.N_iterations_saving_all(delta_time,n_steps,N_iterations)

x = np.linspace(x_min, x_max, 1000)
y = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
plt.scatter(all_positions[0,:,0],all_positions[0,:,1], color='red', marker='o')
for i in range(1,N_iterations):
  plt.scatter(all_positions[i,:,0],all_positions[i,:,1], c=hmc_nm.likelihood(all_positions[i,:,0],all_positions[i,:,1]), cmap='viridis',vmin=vmin, vmax=vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-20,20)
plt.title('Likelihood Function Colormap')
plt.grid(True)
plt.savefig('prior_posterior_mh_hmc.png',dpi=500)
plt.show()



plt.figure(figsize=(8, 6))
x = np.linspace(x_min, x_max, 1000)
y = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x, y)
Z = hmc_nm.likelihood(X, Y)
cmap = plt.get_cmap('viridis')
vmin = np.min(Z)
vmax = np.max(Z)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Likelihood')
# Initial positions in red
plt.scatter(all_positions[0,:,0],all_positions[0,:,1], color='black', marker='o', label='Iteration 0')


plt.scatter(all_positions[0,:,0],all_positions[0,:,1], color='purple', marker='.')



# Plot subsequent iterations with colors scaled by iteration number
for i in range(1,N_iterations):
  # The color value for all particles in this iteration is i / N_iterations
  color_value = i / N_iterations
  plt.scatter(all_positions[i,:,0],all_positions[i,:,1],
              c=np.repeat(color_value, all_positions.shape[1]), # Assign the same color value to all particles in iteration i
              cmap='rainbow', # Use a perceptually uniform colormap
              vmin=0.0,      # Min value for color mapping (start of iterations)
              vmax=1.0,      # Max value for color mapping (end of iterations)
              marker='.',    # Smaller marker for individual points
              alpha=0.7,     # Transparency to see overlapping points
              s=50)          # Size of markers

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-20,20)
plt.ylim(-20,20) # Set y-limits for consistency
plt.title('Particle Positions Over Iterations (Color scaled by Iteration Number)')
plt.grid(True)

# Add a colorbar to explain the color scaling
sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=0.0, vmax=1.0))
sm.set_array([]) # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Iteration Progress (i / N_iterations)')

plt.legend() # Display legend for Iteration 0
plt.tight_layout() # Adjust layout
plt.savefig('prior_posterior_evolution_mh_hmc.png',dpi=500)
plt.show()

plt.figure(figsize=(8, 6))
# Initial positions in red
plt.scatter(all_positions[0,:,0],all_positions[0,:,1], color='black', marker='o', label='Iteration 0')
plt.scatter(all_positions[0,:,0],all_positions[0,:,1], color='purple', marker='.')

# Plot subsequent iterations with colors scaled by iteration number
for i in range(1,N_iterations):
  # The color value for all particles in this iteration is i / N_iterations
  color_value = i / N_iterations
  plt.scatter(all_positions[i,:,0],all_positions[i,:,1],
              c=np.repeat(color_value, all_positions.shape[1]), # Assign the same color value to all particles in iteration i
              cmap='plasma', # Use a perceptually uniform colormap
              vmin=0.0,      # Min value for color mapping (start of iterations)
              vmax=1.0,      # Max value for color mapping (end of iterations)
              marker='.',    # Smaller marker for individual points
              alpha=0.7,     # Transparency to see overlapping points
              s=50)          # Size of markers

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-20,20)
plt.ylim(-20,20) # Set y-limits for consistency
plt.title('Particle Positions Over Iterations (Color scaled by Iteration Number)')
plt.grid(True)

# Add a colorbar to explain the color scaling
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0.0, vmax=1.0))
sm.set_array([]) # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Iteration Progress (i / N_iterations)')

plt.legend() # Display legend for Iteration 0
plt.tight_layout() # Adjust layout
plt.savefig('nocolormap_prior_posterior_evolution.png',dpi=500)
plt.show()