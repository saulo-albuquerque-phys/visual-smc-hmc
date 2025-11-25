import numpy as np
import matplotlib.pyplot as plt

from particle_memory import particle

class HMC_one_mode:
    """
    A simple Sequential Monte Carlo (SMC) / tempering implementation for a 2D parameter space (x,y).

    Notes / assumptions:
    - Particles are represented as arrays with shape (2, N): samples[0] = x's, samples[1] = y's.
    - Tempering: importance weights are computed as exp((beta_current - beta_prev) * logL).
    - Resampling uses systematic resampling (low variance).
    - Metropolis-Hastings proposals are Gaussian location perturbations (per-particle).
    """

    def __init__(self, prior_ranges_x, prior_ranges_y, likelihood_parameters,
                 ess_rate, n_particles, rng_seed=None):
        assert len(prior_ranges_x) == 2 and len(prior_ranges_y) == 2
        self.prior_range_x_left, self.prior_range_x_right = prior_ranges_x
        self.prior_range_y_down, self.prior_range_y_up = prior_ranges_y

        self.likelihood_parameters = likelihood_parameters
        self.ess_rate = ess_rate
        self.n_particles = int(n_particles)

        self.particles=None

        # unpack likelihood parameters (expects keys to exist)
        self.x1 = likelihood_parameters['x1']
        self.x2 = likelihood_parameters['x2']
        self.y1 = likelihood_parameters['y1']
        self.y2 = likelihood_parameters['y2']
        self.alpha_x1 = max(likelihood_parameters['alpha_x1'], 1e-12)
        self.alpha_x2 = max(likelihood_parameters['alpha_x2'], 1e-12)
        self.alpha_y1 = max(likelihood_parameters['alpha_y1'], 1e-12)
        self.alpha_y2 = max(likelihood_parameters['alpha_y2'], 1e-12)

        # RNG (use a single generator for reproducibility)
        self.rng = np.random.default_rng(rng_seed)

        # runtime fields (populated by prior())
        self.prior_samples = None
        self.prior_temperature_par = None
        self.unnormalized_prior_weights = None
        self.normalized_prior_weights = None
        self.normalized_prior_evidence = None
        self.prior_effective_sample_size = None

    # ----------------------------
    # likelihoods
    # ----------------------------
    def likelihood(self, x, y):
        """
        Vectorized likelihood (mixture of two Gaussians with given parameters).
        x and y can be scalars or numpy arrays (must broadcast together).
        Guarded to avoid zero or negative values (floor).
        """
        ax1=self.alpha_x1
        ay1=self.alpha_y1


        term1_norm = 10.0 / (ax1**2 + ay1**2)
        #term2_norm = 10.0 / (ax2**2 + ay2**2)

        term1_exp = np.exp(-((1.0/ax1)**2) * (x - self.x1)**2 - ((1.0/ay1)**2) * (y - self.y1)**2)
        #term2_exp = np.exp(-((1.0/ax2)**2) * (x - self.x2)**2 - ((1.0/ay2)**2) * (y - self.y2)**2)

        L = term1_norm * term1_exp
        #L = term1_norm * term1_exp + term2_norm * term2_exp

        # guard against exact zeros for log stability
        L = np.maximum(L, 1e-300)
        return L

    def log_likelihood(self, x, y):
        ax1=self.alpha_x1
        ay1=self.alpha_y1
        term1_norm = 10.0 / (ax1**2 + ay1**2)
        #term2_norm = 10.0 / (ax2**2 + ay2**2)

        term1_exp = np.exp(-((1.0/ax1)**2) * (x - self.x1)**2 - ((1.0/ay1)**2) * (y - self.y1)**2)
        #term2_exp = np.exp(-((1.0/ax2)**2) * (x - self.x2)**2 - ((1.0/ay2)**2) * (y - self.y2)**2]

        L = term1_norm * term1_exp
        #L = term1_norm * term1_exp + term2_norm * term2_exp

        logL=np.log(term1_norm) -((1.0/ax1)**2) * (x - self.x1)**2 - ((1.0/ay1)**2) * (y - self.y1)**2
        return logL

    def derivative_x_log_likelihood(self, x, y):
        ax1=self.alpha_x1
        ay1=self.alpha_y1
        derivative_x_log_likelihood= -((1.0/ax1)**2)*2*(x - self.x1)
        return derivative_x_log_likelihood

    def derivative_y_log_likelihood(self, x, y):
        ax1=self.alpha_x1
        ay1=self.alpha_y1
        derivative_y_log_likelihood= -((1.0/ay1)**2)*2*(y - self.y1)
        return derivative_y_log_likelihood

    def gradient_log_likelihood(self, x, y):
        return np.array([self.derivative_x_log_likelihood(x, y),self.derivative_y_log_likelihood(x, y)])

    # ----------------------------
    # priors / initialization
    # ----------------------------
    def _prior_samples(self):
        x_samples = self.rng.uniform(self.prior_range_x_left, self.prior_range_x_right, self.n_particles)
        y_samples = self.rng.uniform(self.prior_range_y_down, self.prior_range_y_up, self.n_particles)
        return np.array([x_samples, y_samples])  # shape (2, N)

    def prior(self):
        """Initialize prior particles, weights, ESS, evidence baseline."""
        prior_samples = self._prior_samples()
        self.prior_samples = prior_samples
        self.prior_temperature_par = 0.0

        # unnormalized importance weights for initial temperature step:
        # current_temperature_par = prior_temperature_par, previous_temperature_par = 0.0
        self.unnormalized_prior_weights = self.unnormalized_importance_weights(
            self.prior_samples, self.prior_temperature_par, 0.0)
        # normalized weights
        self.normalized_prior_weights = self.unnormalized_prior_weights / np.sum(self.unnormalized_prior_weights)
        self.normalized_prior_evidence = 1.0  # baseline
        self.prior_effective_sample_size = self.ESS(self.unnormalized_prior_weights)
        self.prior_particle_distribution_per_state = np.ones(self.n_particles) / self.n_particles



    ## setting particles

    def set_particles_initial_from_dict(self,dict_particles):
        particles=[]
        n_particles=len(dict_particles.keys())
        for i,key in enumerate(dict_particles.keys()):
            particle=dict_particles[key]
            particles.append(particle)
            position_particle=np.array([particle.x,particle.y])
            momentum_particle=np.array([particle.px,particle.py])
            particle._attibute_mass(np.identity(2))
            particle._atribute_likelihood(self.likelihood(position_particle[0],position_particle[1]))
            particle._atribute_log_likelihood(self.log_likelihood(position_particle[0],position_particle[1]))
            particle._atribute_gradient_log_likelihood(self.gradient_log_likelihood(position_particle[0],position_particle[1]))
        self.particles=particles

    def set_particles_initial_from_list(self,list_particles):
        particles=[]
        for i, part in enumerate(list_particles):
            particle=part
            position_particle=np.array([particle.x,particle.y])
            momentum_particle=np.array([particle.px,particle.py])
            particle._attibute_mass(np.identity(2))
            particle._atribute_likelihood(self.likelihood(position_particle[0],position_particle[1]))
            particle._atribute_log_likelihood(self.log_likelihood(position_particle[0],position_particle[1]))
            particle._atribute_gradient_log_likelihood(self.gradient_log_likelihood(position_particle[0],position_particle[1]))
            particles.append(particle)
        self.particles=particles

    def randomly_generate_particles_from_prior(self, N_particles, factor_momentum=1.):
        prior_samples = self._prior_samples()
        self.prior_samples = prior_samples
        px_samples = self.rng.uniform(factor_momentum*self.prior_range_x_left, factor_momentum*self.prior_range_x_right, self.n_particles)
        py_samples = self.rng.uniform(factor_momentum*self.prior_range_y_down, factor_momentum*self.prior_range_y_up, self.n_particles)
        p_samples=np.array([px_samples, py_samples])
        initial_location_particles=self.prior_samples
        initial_momentum_particules=p_samples
        particles=[]
        for i in range(N_particles):
            position_particle=np.array([initial_location_particles[0,i],initial_location_particles[1,i]])
            momentum_particle=np.array([initial_momentum_particules[0,i],initial_momentum_particules[1,i]])
            particlee=particle(position_particle,momentum_particle)
            identity=np.identity(2)
            particlee._atribute_mass(identity)
            particlee._atribute_likelihood(self.likelihood(position_particle[0],position_particle[1]))
            particlee._atribute_log_likelihood(self.log_likelihood(position_particle[0],position_particle[1]))
            particlee._atribute_gradient_log_likelihood(self.gradient_log_likelihood(position_particle[0],position_particle[1]))
            particles.append(particlee)
        self.particles=particles

    def one_leapfrog_step_looped_non_paralellized(self, delta_time):
        for particle in self.particles:
            particle.first_half_accelerate(delta_time,particle.gradient_log_likelihood)
            particle.move(delta_time)
            new_position=particle.position
            new_value_likelihood=self.likelihood(new_position[0],new_position[1])
            new_value_log_likelihood=self.log_likelihood(new_position[0],new_position[1])
            new_value_gradient_log_likelihood=self.gradient_log_likelihood(new_position[0],new_position[1])
            particle._atribute_likelihood(new_value_likelihood)
            particle._atribute_log_likelihood(new_value_log_likelihood)
            particle._atribute_gradient_log_likelihood(new_value_gradient_log_likelihood)
            particle.second_half_accelerate(delta_time,new_value_gradient_log_likelihood)
            new_momentum=particle.momentum

    def N_leapfrog_steps_looped_non_paralellized(self, delta_time, n_steps):
        for i in range(n_steps):
            self.one_leapfrog_step_looped_non_paralellized(delta_time)

    def metropolis_hasting_test(self,delta_time,n_steps):
        self.N_leapfrog_steps_looped_non_paralellized(delta_time,n_steps)
        final_positions=np.zeros((len(self.particles),2))
        for i, particle in enumerate(self.particles):
            log_likelihood=particle.historic_log_likelihood
            positions=particle.historic_position
            position_initial=positions[0]
            position_final=positions[-1]
            log_likelihood_initial=log_likelihood[0]
            log_likelihood_final=log_likelihood[-1]
            # log acceptance ratio
            log_r = log_likelihood_final - log_likelihood_initial
            # compute acceptance probabilities safely: min(1, exp(log_r))
            # use np.minimum to avoid overflow when log_r is large positive
            accept_prob = np.exp(np.minimum(log_r, 0.0))  # equals min(1, exp(log_r))
            u = self.rng.random()
            accept = (u < accept_prob)
            if accept:
                self.particles[i]._atribute_position(position_final)
                final_positions[i]=position_final
            else:
                self.particles[i]._atribute_position(position_initial)
                final_positions[i]=position_initial
        return final_positions


    def calculate_mass(self):
        all_particles_locations=np.zeros((self.n_particles,2))
        for i,particle in enumerate(self.particles):
            position_particle=np.array([particle.x,particle.y])
            all_particles_locations[i,:]=position_particle
        covariance_R=np.cov(np.array(all_particles_locations).T)
        print("covariance_R",covariance_R)
        #warmed up estimated empirical mass
        Mass=covariance_R
        return Mass

    def update_masses(self):
        Mass=self.calculate_mass()
        for particle in self.particles:
            particle._atribute_mass(Mass)

    def sample_momentum_gaussian(self):
        Mass=self.calculate_mass()
        L = np.linalg.cholesky(Mass)
        for particle in self.particles:
            z=np.random.randn(2)
            p=L @ z
            particle._atribute_momentum(p)

    def first_iteration(self, delta_time, n_steps):
        return self.metropolis_hasting_test(delta_time,n_steps)

    def iteration(self,delta_time,n_steps):
        self.update_masses()
        self.sample_momentum_gaussian()
        self.metropolis_hasting_test(delta_time,n_steps)

    def N_iterations(self,delta_time,n_steps,N_iterations):
        all_final_samples=np.zeros((N_iterations,len(self.particles),2))
        for i in range(N_iterations):
          self.update_masses()
          self.sample_momentum_gaussian()
          samples=self.metropolis_hasting_test(delta_time,n_steps)
          all_final_samples[i,:,:]=samples
        return all_final_samples

