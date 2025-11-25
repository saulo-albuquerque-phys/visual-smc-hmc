import numpy as np
import matplotlib.pyplot as plt
from SMC_particle_no_memory.smc_no_particle_memory import HMC_no_historic


x1,y1=-5,5
x2,y2=5,-5
alpha_x1,alpha_y1=8,3
alpha_x2,alpha_y2=2,3

PRIOR_ranges_X=(-20,20)
PRIOR_ranges_Y=(-20,20)
likelihood_parameters={'x1':x1,'x2':x2,'y1':y1,'y2':y2,'alpha_x1':alpha_x1,'alpha_x2':alpha_x2,'alpha_y1':alpha_y1, 'alpha_y2':alpha_y2}
ess_rate=0.9
n_particles=1000

x_min,x_max=PRIOR_ranges_X
y_min,y_max=PRIOR_ranges_Y




hmc_nm=HMC_only_one_mode_only_one_MH_step(PRIOR_ranges_X,PRIOR_ranges_Y,likelihood_parameters,ess_rate,n_particles)

x = np.linspace(x_min, x_max, 1000)
y = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x, y)

Z = hmc_nm.likelihood(X, Y)

cmap = plt.get_cmap('viridis')
vmin = np.min(Z)
vmax = np.max(Z)



iterations=5

samples_moved, samples_resampled, temperature, weights, indices, evidence, results=hmc_nm.iterations_saving(nsteps_MH=2, delta_time=0.001, n_steps=1000, n_iterations=iterations )

results

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML # For displaying in notebooks

# Define plot range for the scattered points
x_min, x_max = PRIOR_ranges_X
y_min, y_max = PRIOR_ranges_Y

# Create figure and axes for the animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# Store frames for the animation
frames = []

# Determine the total number of iterations from the stored results
num_iterations = len(results['temperature_par'])

# Get a colorbar handle for consistent scaling
sample_likelihoods = np.concatenate([hmc_nm.likelihood(results['samples_resampled'][i, 0, :], results['samples_resampled'][i, 1, :]) for i in range(num_iterations)] + \
                                   [hmc_nm.likelihood(results['samples_moved'][i, 0, :], results['samples_moved'][i, 1, :]) for i in range(num_iterations)])
vmin = np.min(sample_likelihoods)
vmax = np.max(sample_likelihoods)

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # empty array to suppress warning
fig.colorbar(sm, ax=ax, label='Likelihood')


for i in range(num_iterations):
    current_temp = results['temperature_par'][i]

    # Frame for samples_resampled
    resampled_x = results['samples_resampled'][i, 0, :]
    resampled_y = results['samples_resampled'][i, 1, :]
    likelihood_resampled = hmc_nm.likelihood(resampled_x, resampled_y)

    # Use the same colormap and normalization for consistent coloring
    sc_resampled = ax.scatter(resampled_x, resampled_y,
                              c=likelihood_resampled, cmap=cmap, norm=norm, s=10,
                              edgecolor='none', alpha=0.7)
    # Title for resampling step
    title_text_resampled = ax.text(0.5, 1.05, 'RESAMPLING STEP',
                                   horizontalalignment='center', verticalalignment='bottom',
                                   transform=ax.transAxes, fontsize=14, fontweight='bold')
    # Beta parameter at the bottom
    beta_text_resampled = ax.text(0.5, 0.02, f'Beta = {current_temp:.3f}',
                                  horizontalalignment='center', verticalalignment='bottom',
                                  transform=ax.transAxes, fontsize=12)
    frames.append([sc_resampled, title_text_resampled, beta_text_resampled])

    # Frame for samples_moved
    moved_x = results['samples_moved'][i, 0, :]
    moved_y = results['samples_moved'][i, 1, :]
    likelihood_moved = hmc_nm.likelihood(moved_x, moved_y)

    # Use the same colormap and normalization for consistent coloring
    sc_moved = ax.scatter(moved_x, moved_y,
                          c=likelihood_moved, cmap=cmap, norm=norm, s=10,
                          edgecolor='none', alpha=0.7)
    # Title for moving step
    title_text_moved = ax.text(0.5, 1.05, 'MOVING STEP',
                              horizontalalignment='center', verticalalignment='bottom',
                              transform=ax.transAxes, fontsize=14, fontweight='bold')
    # Beta parameter at the bottom
    beta_text_moved = ax.text(0.5, 0.02, f'Beta = {current_temp:.3f}',
                              horizontalalignment='center', verticalalignment='bottom',
                              transform=ax.transAxes, fontsize=12)
    frames.append([sc_moved, title_text_moved, beta_text_moved])

# Create the animation
ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=1000)

# Save the animation
try:
    ani.save('smc_hmc_animation_enhanced.gif', writer='imagemagick', fps=2)
except ValueError:
    print("ImageMagick writer not found, falling back to Pillow writer.")
    ani.save('hmc_nm_animation_enhanced.gif', writer='pillow', fps=2)

plt.close(fig)
print("Animation saved as 'smc_hmc_animation_enhanced.gif'")