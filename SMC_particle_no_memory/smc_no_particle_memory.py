import numpy as np
import matplotlib.pyplot as plt

from particle_no_memory import particle_no_historic

class HMC_no_historic:
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

    # FIRST ORDER DERIVATIVES

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

    # Second Order Derivatives

    def derivative_x_derivative_x_log_likelihood(self,x,y):
      ax1=self.alpha_x1
      ay1=self.alpha_y1
      derivative_x_derivative_x_log_likelihood= -2*((1.0/ax1)**2)
      return derivative_x_derivative_x_log_likelihood

    def derivative_y_derivative_y_log_likelihood(self,x,y):
      ax1=self.alpha_x1
      ay1=self.alpha_y1
      derivative_y_derivative_y_log_likelihood= -2*((1.0/ay1)**2)
      return derivative_y_derivative_y_log_likelihood

    def hessian_matrix(self,x,y):
      hessian_matrix=np.array([[self.derivative_x_derivative_x_log_likelihood(x,y),0],[0, self.derivative_y_derivative_y_log_likelihood(x,y)]])
      return hessian_matrix

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


    # ----------------------------
    # importance weights & ESS
    # ----------------------------
    def unnormalized_importance_weights(self, samples, current_temperature_par, previous_temperature_par):
        """
        Given samples (shape (2,N)) and two temperatures, return unnormalized weights:
            w_i = exp((beta_cur - beta_prev) * logL(x_i))
        """
        x_samples = samples[0]
        y_samples = samples[1]
        logL = self.log_likelihood(x_samples, y_samples)
        # safe exponent: use maximum lower bound on logL so we don't overflow to inf (logL may be large negative)
        delta = (current_temperature_par - previous_temperature_par)
        w = np.exp(delta * logL)
        # avoid all zeros
        w = np.maximum(w, 1e-300)
        return w

    def ESS(self, unnormalized_weights):
        """Effective sample size given unnormalized weights: (sum w)^2 / sum w^2"""
        w = np.asarray(unnormalized_weights)
        denom = np.sum(w**2)
        if denom <= 0:
            return 0.0
        return (np.sum(w)**2) / denom

    def ESS_of_temp_par(self, logL, current_temperature_par, previous_temperature_par):
        """
        Compute ESS when moving from previous_temperature_par to current_temperature_par,
        based on logL array (shape (N,)).
        """
        delta = (current_temperature_par - previous_temperature_par)
        # compute unnormalized w then normalize
        logw = delta * logL
        # subtract max for numerical stability before exponentiating
        logw = logw - np.max(logw)
        w = np.exp(logw)
        w_sum = np.sum(w)
        if w_sum <= 0:
            return 0.0
        w = w / w_sum
        return 1.0 / np.sum(w**2)

    # ----------------------------
    # resampling
    # ----------------------------
    def systematic_resample(self, samples, weights):
        """
        Systematic resampling.
        samples: shape (2, N)
        weights: may be normalized or unnormalized; we'll normalize inside.
        Returns: new_samples (2,N), new_weights (N,), indices (N,) referencing original particles
        """
        weights = np.asarray(weights, dtype=float)
        s = np.sum(weights)
        if s <= 0:
            # fallback to uniform if weights bad
            weights = np.ones(self.n_particles) / self.n_particles
        else:
            weights = weights / s

        N = len(weights)
        # positions uniformly spaced with random offset in [0,1/N)
        u0 = self.rng.random()
        positions = (u0 + np.arange(N)) / N

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # numerical safety
        indices = np.searchsorted(cumulative_sum, positions, side='right')

        # vectorized selection
        new_samples = samples[:, indices]   # shape (2, N)
        new_weights = np.ones(N) / N
        return new_samples, new_weights, indices



    ## -----------------------------------------------------------------
    ## HMC SECTION
    ## -----------------------------------------------------------------


    ## setting particles from given dictionary or lists

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

    # saving particles positions and likelihoods

    def saving_particles_positions_and_log_likelihood(self):
        particle_positions=np.zeros((len(self.particles),2))
        particle_log_likelihood=np.zeros(len(self.particles))
        for i, particle in enumerate(self.particles):
            particle_positions[i,:]=particle.position
            particle_log_likelihood[i]=particle.log_likelihood
        return particle_positions, particle_log_likelihood

    #Leapfrog HMC steps

    def one_leapfrog_step_looped_non_paralellized(self, delta_time):
        for particle in self.particles:
            particle.half_accelerate(delta_time,particle.gradient_log_likelihood)
            particle.move(delta_time)
            new_position=particle.position
            new_value_likelihood=self.likelihood(new_position[0],new_position[1])
            new_value_log_likelihood=self.log_likelihood(new_position[0],new_position[1])
            new_value_gradient_log_likelihood=self.gradient_log_likelihood(new_position[0],new_position[1])
            particle._atribute_likelihood(new_value_likelihood)
            particle._atribute_log_likelihood(new_value_log_likelihood)
            particle._atribute_gradient_log_likelihood(new_value_gradient_log_likelihood)
            particle.half_accelerate(delta_time,new_value_gradient_log_likelihood)
            new_momentum=particle.momentum

    def N_leapfrog_steps_looped_non_paralellized(self, delta_time, n_steps):
        for i in range(n_steps):
            self.one_leapfrog_step_looped_non_paralellized(delta_time)

    # METROPOLIS HASTING with HMC

    def metropolis_hasting_test(self,delta_time,n_steps):
        initial_position_particles,initial_particle_log_likelihood=self.saving_particles_positions_and_log_likelihood()
        self.N_leapfrog_steps_looped_non_paralellized(delta_time,n_steps)
        final_position_particles,final_particle_log_likelihood=self.saving_particles_positions_and_log_likelihood()
        log_r=final_particle_log_likelihood-initial_particle_log_likelihood
        accept_prob = np.exp(np.minimum(log_r, 0.0))
        u = self.rng.random(len(self.particles))
        accept = (u < accept_prob)
        final_positions=np.zeros((len(self.particles),2))
        for i, particle in enumerate(self.particles):
            if accept[i]:
                self.particles[i]._atribute_position(final_position_particles[i])
                final_positions[i]=final_position_particles[i]
            else:
                self.particles[i]._atribute_position(initial_position_particles[i])
                final_positions[i]=initial_position_particles[i]
        return final_positions

    # HESSIAN MATRIX for MASSES

    def update_hessian_masses(self):
        for i,particle in enumerate(self.particles):
            position_particle=np.array([particle.x,particle.y])
            hessian=self.hessian_matrix(position_particle[0],position_particle[1])
            particle._atribute_mass(hessian)


    def update_masses_sample_momentum(self):
        print(f"updating hessian masses and sampling momentum from masses")
        for i,particle in enumerate(self.particles):
            position_particle=np.array([particle.x,particle.y])
            hessian=self.hessian_matrix(position_particle[0],position_particle[1])
            mass=-hessian
            particle._atribute_mass(mass)
            z=np.random.randn(2)
            L = np.linalg.cholesky(mass)
            p=L @ z
            momentum_particle=np.array([p[0],p[1]])
            particle._atribute_momentum(momentum_particle)

    # INITIALIZING PARTICLE POSITION AND MOMENTUM FROM PRIOR

    def randomly_generate_particle_position_from_prior_and_momentum_from_riemann(self, N_particles):
        print(f"generating {N_particles} particles from prior")
        prior_samples = self._prior_samples()
        self.prior_samples = prior_samples
        initial_location_particles=self.prior_samples
        particles=[]
        for i in range(N_particles):
            position_particle=np.array([initial_location_particles[0,i],initial_location_particles[1,i]])
            hessian=self.hessian_matrix(position_particle[0],position_particle[1])
            mass=-hessian
            z=np.random.randn(2)
            try:
              L = np.linalg.cholesky(mass)
            except:
              #print(f'negative mass for {position_particle}')
              L = np.identity(2)
            p=L @ z
            momentum_particle=np.array([p[0],p[1]])
            particlee=particle_no_historic(position_particle,momentum_particle)
            particlee._atribute_mass(mass)
            particlee._atribute_likelihood(self.likelihood(position_particle[0],position_particle[1]))
            particlee._atribute_log_likelihood(self.log_likelihood(position_particle[0],position_particle[1]))
            particlee._atribute_gradient_log_likelihood(self.gradient_log_likelihood(position_particle[0],position_particle[1]))
            particles.append(particlee)
        self.particles=particles

    # ITERATION

    def iteration_HMC(self,delta_time,n_steps):
        N_particles=self.n_particles
        if self.particles is None:
          self.randomly_generate_particle_position_from_prior_and_momentum_from_riemann(N_particles)
          initial_positions,initial_likelihood=self.saving_particles_positions_and_log_likelihood()
        else:
          initial_positions,initial_likelihood=self.saving_particles_positions_and_log_likelihood()
          self.update_masses_sample_momentum()
        self.metropolis_hasting_test(delta_time,n_steps)
        final_positions,final_likelihood=self.saving_particles_positions_and_log_likelihood()
        return initial_positions,initial_likelihood,final_positions,final_likelihood

    def N_iterations_saving_all(self,delta_time,n_steps, N_iterations):
        all_positions=np.zeros((N_iterations,self.n_particles,2))
        all_likelihood=np.zeros((N_iterations,self.n_particles))
        first_positions, first_likelihood,second_positions,second_likelihood=self.iteration_HMC(delta_time,n_steps)
        all_positions[0,:,:]=first_positions
        all_likelihood[0,:]=first_likelihood
        all_positions[1,:,:]=second_positions
        all_likelihood[1,:]=second_likelihood
        for i in range(2,N_iterations):
          _,_,positions,likelihood=self.iteration_HMC(delta_time,n_steps)
          all_positions[i,:,:]=positions
          all_likelihood[i,:]=likelihood
        return all_positions, all_likelihood

    def N_iterations_saving_only_start_end(self,delta_time,n_steps, N_iterations):
        all_positions=np.zeros((2,self.n_particles,2))
        all_likelihood=np.zeros((2,self.n_particles))
        first_positions, first_likelihood,second_positions,second_likelihood=self.iteration_HMC(delta_time,n_steps)
        all_positions[0,:,:]=first_positions
        all_likelihood[0,:]=first_likelihood
        all_positions[1,:,:]=second_positions
        all_likelihood[1,:]=second_likelihood
        for i in range(2,N_iterations):
          _,_,positions,likelihood=self.iteration_HMC(delta_time,n_steps)
          all_positions[1,:,:]=positions
          all_likelihood[1,:]=likelihood
        return all_positions, all_likelihood




    # ----------------------------
    # beta finding (tempering schedule)
    # ----------------------------
    def find_next_beta(self, samples, previous_temperature_par, tolerance=1e-4, max_iter=60):
        """
        Binary search for next beta in (previous_beta, 1.0] such that ESS(beta)/N ≈ ess_rate.
        Returns new_beta.
        """
        x_samples = samples[0]
        y_samples = samples[1]
        logL = self.log_likelihood(x_samples, y_samples)

        lo, hi = float(previous_temperature_par), 1.0
        target_ess = self.ess_rate * self.n_particles

        # If already full temperature satisfies target, return hi
        ess_hi = self.ESS_of_temp_par(logL, hi, previous_temperature_par)
        if ess_hi >= target_ess:
            return hi

        # Binary search (lo, hi)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            ess_mid = self.ESS_of_temp_par(logL, mid, previous_temperature_par)
            if ess_mid < target_ess:
                hi = mid
            else:
                lo = mid
            if hi - lo <= tolerance:
                break
        return 0.5 * (lo + hi)

    # ----------------------------
    # high-level iterations
    # ----------------------------
    def first_iteration(self, nsteps_MH=1, delta_time=0.001, n_steps=1000):
        """
        Initialize from prior, compute initial evidence ratio, resample, do MH moves,
        then find next beta.
        Returns: iteration_index (int), new_samples, new_temperature_par, new_weights, indices, evidence_first_iteration
        """
        if self.prior_samples is None:
            raise RuntimeError("Call .prior() before running iterations()")

        samples = self.prior_samples
        temperature_par = self.prior_temperature_par
        importance_weights_first_iteration = self.normalized_prior_weights
        unnormalized_importance_weights_first_iteration = self.unnormalized_prior_weights

        # evidence update (ratio)
        ratio_evidence = (1.0 / self.n_particles) * np.sum(unnormalized_importance_weights_first_iteration)
        evidence_first_iteration = ratio_evidence * self.normalized_prior_evidence

        # resample according to normalized prior weights
        new_samples, new_weights, indices = self.systematic_resample(samples, importance_weights_first_iteration)

        # local moves (MH HMC )
        new_samples_moved_0,_ = self.N_iterations_saving_only_start_end(delta_time,n_steps, nsteps_MH)
        new_samples_moved_x=new_samples_moved_0[1,:,0]
        new_samples_moved_y=new_samples_moved_0[1,:,1]
        new_samples_moved=np.array([new_samples_moved_x,new_samples_moved_y])

        # find next temperature beta
        new_temperature_par = self.find_next_beta(new_samples_moved, temperature_par)

        return 1, new_samples_moved, new_samples, new_temperature_par, new_weights, indices, evidence_first_iteration

    def iteration(self, nsteps_MH, delta_time, n_steps, iteration, samples_moved,samples_resampled, temperature_par, unnormalized_importance_weights, indices, evidence):
        """
        One SMC tempering iteration. Assumes unnormalized_importance_weights correspond to the current
        temperature step (i.e., weights that integrate the ratio for evidence update).
        """
        # evidence update
        ratio_evidence = (1.0 / self.n_particles) * np.sum(unnormalized_importance_weights)
        evidence = ratio_evidence * evidence

        # normalize weights for resampling
        importance_weights = unnormalized_importance_weights / np.sum(unnormalized_importance_weights)

        # resample
        new_samples, new_weights, indices = self.systematic_resample(samples_moved, importance_weights)

        # local moves (MH HMC )
        new_samples_moved_0,_ = self.N_iterations_saving_only_start_end(delta_time,n_steps, nsteps_MH)
        new_samples_moved_x=new_samples_moved_0[1,:,0]
        new_samples_moved_y=new_samples_moved_0[1,:,1]
        new_samples_moved=np.array([new_samples_moved_x,new_samples_moved_y])


        # find next temperature (beta)
        new_temperature_par = self.find_next_beta(new_samples_moved, temperature_par)

        return iteration + 1, new_samples_moved, new_samples, new_temperature_par, new_weights, indices, evidence

    def iterations(self,  nsteps_MH=1, delta_time=0.001, n_steps=1000, n_iterations=100, verbose=True):
        """
        Run many iterations starting from prior().
        Returns final samples, temperature, weights, indices, evidence.
        """
        # ensure prior initialized
        if self.prior_samples is None:
            self.prior()

        iteration, samples_moved, samples_resampled, temperature, weights, indices, evidence = self.first_iteration(n_steps_MH, delta_time, nsteps)
        if verbose:
            print(f"iteration: {iteration}, temperature: {temperature:.6f}, evidence: {evidence:.6e}")

        for i in range(n_iterations):
            # compute unnormalized weights for moving from current temperature to candidate next temperature:
            # The find_next_beta uses the samples after MH moves to propose the next beta, but the
            # unnormalized importance weights we need for evidence are:
            #    w_i = exp((beta_new - beta_old) * logL(x_i))
            # Here we choose beta_new by calling find_next_beta on samples,
            # so we must compute unnormalized weights consistent with that beta.
            # To keep the loop consistent, compute logL on current samples and then find beta.
            logL = self.log_likelihood(samples_moved[0], samples_moved[1])
            beta_candidate = self.find_next_beta(samples_moved, temperature)
            unnormalized_weights = np.exp((beta_candidate - temperature) * logL)
            unnormalized_weights = np.maximum(unnormalized_weights, 1e-300)

            # proceed with iteration using those unnormalized weights
            iteration, samples_moved, samples_resampled, temperature, weights, indices, evidence = self.iteration(
                n_steps_MH, delta_time, nsteps, iteration, samples_moved, samples_resampled, temperature, unnormalized_weights, indices, evidence)

            if verbose:
                print(f"iteration: {iteration}, temperature: {temperature:.6f}, evidence: {evidence:.6e}")

            # stop early if temperature reached 1.0 (full target)
            if temperature >= 1.0 - 1e-12:
                break

        return samples_moved, temperature, weights, indices, evidence

    def iterations_saving(self, nsteps_MH=1, delta_time=0.001, n_steps=1000, n_iterations=100, verbose=True):
        """
        Run many iterations starting from prior().
        Returns final samples, temperature, weights, indices, evidence.
        Saves everything into dictionary
        """
        # dictionary
        results={}

        #ndarrays
        samples_moved_all=np.zeros((n_iterations,2,self.n_particles))
        samples_resampled_all=np.zeros((n_iterations,2,self.n_particles))
        temperature_par_all=np.zeros(n_iterations)
        weights_all=np.zeros((n_iterations,self.n_particles))


        # ensure prior initialized
        if self.prior_samples is None:
            self.prior()
        samples_moved_all[0,:,:]=self.prior_samples
        samples_resampled_all[0,:,:]=self.prior_samples
        temperature_par_all[0]=self.prior_temperature_par
        weights_all[0,:]=self.normalized_prior_weights

        iteration, samples_moved, samples_resampled, temperature, weights, indices, evidence = self.first_iteration(nsteps_MH, delta_time, n_steps)
        if verbose:
            print(f"iteration: {iteration}, temperature: {temperature:.6f}, evidence: {evidence:.6e}")
        samples_moved_all[1,:,:]=samples_moved
        samples_resampled_all[1,:,:]=samples_resampled
        temperature_par_all[1]=temperature
        weights_all[1,:]=weights



        for i in range(n_iterations-2):
            # compute unnormalized weights for moving from current temperature to candidate next temperature:
            # The find_next_beta uses the samples after MH moves to propose the next beta, but the
            # unnormalized importance weights we need for evidence are:
            #    w_i = exp((beta_new - beta_old) * logL(x_i))
            # Here we choose beta_new by calling find_next_beta on samples,
            # so we must compute unnormalized weights consistent with that beta.
            # To keep the loop consistent, compute logL on current samples and then find beta.
            logL = self.log_likelihood(samples_moved[0], samples_moved[1])
            beta_candidate = self.find_next_beta(samples_moved, temperature)
            unnormalized_weights = np.exp((beta_candidate - temperature) * logL)
            unnormalized_weights = np.maximum(unnormalized_weights, 1e-300)

            # proceed with iteration using those unnormalized weights
            iteration, samples_moved, samples_resampled, temperature, weights, indices, evidence = self.iteration(
               nsteps_MH, delta_time, n_steps, iteration, samples_moved, samples_resampled, temperature, unnormalized_weights, indices, evidence)
            samples_moved_all[iteration,:,:]=samples_moved
            samples_resampled_all[iteration,:,:]=samples_resampled
            temperature_par_all[iteration]=temperature
            weights_all[iteration,:]=weights

            if verbose:
                print(f"iteration: {iteration}, temperature: {temperature:.6f}, evidence: {evidence:.6e}")

            # stop early if temperature reached 1.0 (full target)
            if temperature >= 1.0 - 1e-12:
                break
        results['samples_moved']=samples_moved_all
        results['samples_resampled']=samples_resampled_all
        results['temperature_par']=temperature_par_all
        results['weights']=weights_all
        return samples_moved, samples_resampled, temperature, weights, indices, evidence, results


