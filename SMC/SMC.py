import numpy as np
import matplotlib.pyplot as plt

class SMC():
  def __init__(self,prior_ranges_x,prior_ranges_y,likelihood_parameters,ess_rate,n_particles):
    assert len(prior_ranges_x)==2 and len(prior_ranges_y)==2
    self.prior_range_x_left,self.prior_range_x_right = prior_ranges_x
    self.prior_range_y_down,self.prior_range_y_up = prior_ranges_y
    self.likelihood_parameters = likelihood_parameters
    self.ess_rate = ess_rate
    self.n_particles = n_particles
    self.x1 = self.likelihood_parameters['x1']
    self.x2 = self.likelihood_parameters['x2']
    self.y1 = self.likelihood_parameters['y1']
    self.y2 = self.likelihood_parameters['y2']
    self.alpha_x1 = self.likelihood_parameters['alpha_x1']
    self.alpha_x2 = self.likelihood_parameters['alpha_x2']
    self.alpha_y1 = self.likelihood_parameters['alpha_y1']
    self.alpha_y2 = self.likelihood_parameters['alpha_y2']
    self.rng=np.random.default_rng()

  def likelihood(self,x,y):
    x1=self.x1
    x2=self.x2
    y1=self.y1
    y2=self.y2
    alpha_x1=self.alpha_x1
    alpha_x2=self.alpha_x2
    alpha_y1=self.alpha_y1
    alpha_y2=self.alpha_y2
    return (10/((alpha_x1)**2+(alpha_y1)**2))*np.exp(-((1/alpha_x1)**2)*(x-x1)**2 - ((1/alpha_y1)**2)*(y-y1)**2)+(10/((alpha_x2)**2+(alpha_y2)**2))*np.exp(- ((1/alpha_x2)**2)*(x-x2)**2 - ((1/alpha_y2)**2)*(y-y2)**2)

  def log_likelihood(self,x,y):
    L=self.likelihood(x,y)
    return np.log(L)

  def _prior_samples(self):
    x_samples = self.rng.uniform(self.prior_range_x_left, self.prior_range_x_right, self.n_particles)
    y_samples = self.rng.uniform(self.prior_range_y_down, self.prior_range_y_up, self.n_particles)
    return np.array([x_samples, y_samples])

  def gaussian_location_perturbation_step(self, samples, cov_x, cov_y):
    x_samples = samples[0]
    y_samples = samples[1]
    x_samples_perturbed = x_samples + self.rng.normal(0, cov_x, self.n_particles)
    y_samples_perturbed = y_samples + self.rng.normal(0, cov_y, self.n_particles)
    return np.array([x_samples_perturbed, y_samples_perturbed])

  def systematic_resample(self, samples, weights):
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    N = len(weights)
    positions = (self.rng.random() + np.arange(N)) / N
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    indices = np.searchsorted(cumulative_sum, positions)
    new_samples = samples[:, indices]   # shape (2, N)
    new_weights = np.ones(N) / N
    return new_samples, new_weights, indices

  def metropolis_hasting_step(self, samples):
    x = samples[0]
    y = samples[1]
    logL_current = self.log_likelihood(x, y)
    proposed = self.gaussian_location_perturbation_step(samples, 0.01, 0.01)
    x_p = proposed[0]; y_p = proposed[1]
    logL_proposed = self.log_likelihood(x_p, y_p)
    log_r = logL_proposed - logL_current
    # avoid overflow: accept_prob = min(1, exp(log_r))
    accept_prob = np.exp(np.minimum(0.0, log_r))
    u = self.rng.random(self.n_particles)
    accept = u < accept_prob
    new_x = np.where(accept, x_p, x)
    new_y = np.where(accept, y_p, y)
    return np.array([new_x, new_y])

  def ESS(self,unnormalized_importance_weights_val):
    ESS=((np.sum(unnormalized_importance_weights_val))**2)/np.sum(unnormalized_importance_weights_val**2)
    return ESS

  def ESS_of_temp_par(self,logL,current_temperature_par,previous_temperature_par):
    w = np.exp((current_temperature_par-previous_temperature_par) * logL)
    w /= np.sum(w)
    return 1.0 / np.sum(w**2)

  def prior(self):
    prior_samples=self._prior_samples()
    self.prior_samples=prior_samples
    self.prior_temperature_par=0.
    self.unnormalized_prior_weights=self.unnormalized_importance_weights(self.prior_samples,self.prior_temperature_par,0.)
    self.normalized_prior_weights=self.unnormalized_prior_weights/np.sum(self.unnormalized_prior_weights)
    self.normalized_prior_evidence=1.
    self.prior_effective_sample_size=self.ESS(self.unnormalized_prior_weights)
    self.prior_particle_distribution_per_state=np.ones(self.n_particles)/self.n_particles

  def systematic_resample(self, samples, weights):
    N = len(weights)
    rng=self.rng
    positions = (rng.random() + np.arange(N)) / N

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  
    indices = np.searchsorted(cumulative_sum, positions)

    
    particles=list(zip(samples[0],samples[1]))
    new_particles = np.array(particles)[indices]
    new_weights = np.ones(N) / N
    new_samples=np.array([new_particles[:,0],new_particles[:,1]])
    return new_samples, new_weights, indices

  def gaussian_location_perturbation_step(self,samples, cov_x,cov_y):
    x_samples=samples[0]
    y_samples=samples[1]
    x_samples_perturbed=x_samples+np.random.normal(0,cov_x,self.n_particles)
    y_samples_perturbed=y_samples+np.random.normal(0,cov_y,self.n_particles)
    return np.array([x_samples_perturbed,y_samples_perturbed])


  def metropolis_hasting_step(self, samples):
    # samples shape: (2, N)
    x = samples[0]
    y = samples[1]

    logL_current = self.log_likelihood(x, y)

    # propose
    proposed = self.gaussian_location_perturbation_step(samples, 0.01, 0.01)
    x_p = proposed[0]
    y_p = proposed[1]
    logL_proposed = self.log_likelihood(x_p, y_p)

    # log acceptance ratio (handle -inf correctly)
    log_r = logL_proposed - logL_current

    # acceptance probabilities
    accept_prob = np.exp(np.minimum(0.0, log_r))  # min(1, exp(log_r)), stable in log space

    u = self.rng.random(self.n_particles)
    accept = u < accept_prob

    # build new samples: choose proposed where accepted else keep current
    new_x = np.where(accept, x_p, x)
    new_y = np.where(accept, y_p, y)

    return np.array([new_x, new_y])

  def metropolis_hasting_various_steps(self,samples,N_steps):
    for i in range(N_steps):
      samples=self.metropolis_hasting_step(samples)
    return samples

  def find_next_beta(self, samples, previous_temperature_par, tolerance=1e-4):
    x_samples=samples[0]
    y_samples=samples[1]
    logL=self.log_likelihood(x_samples,y_samples)
    # Binary search for β s.t. ESS(β)/N ≈ target_ESS_ratio
    lo, hi = previous_temperature_par, 1.0
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if self.ESS_of_temp_par(logL,mid,previous_temperature_par) / self.n_particles < self.ess_rate : # Fixed here
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


  def first_iteration(self, nsteps_MH):
    samples=self.prior_samples
    temperature_par=self.prior_temperature_par
    importance_weights_first_iteration=self.normalized_prior_weights
    unnormalized_importance_weights_first_iteration=self.unnormalized_prior_weights
    ratio_evidence=(1/self.n_particles)*np.sum(unnormalized_importance_weights_first_iteration)
    evidence_first_iteration=ratio_evidence*self.normalized_prior_evidence
    new_samples, new_weights, indices = self.systematic_resample(samples, importance_weights_first_iteration)
    new_samples_moved=self.metropolis_hasting_various_steps(new_samples,nsteps_MH)
    new_temperature_par=self.find_next_beta(new_samples_moved, temperature_par) # Added temperature_par
    return 2,new_samples_moved,new_temperature_par,new_weights,indices,evidence_first_iteration

  def iteration(self,nsteps_MH,iteration,samples,temperature_par, unnormalized_importance_weights, indices, evidence):
    ratio_evidence=(1/self.n_particles)*np.sum(unnormalized_importance_weights)
    evidence=ratio_evidence*evidence
    importance_weights=unnormalized_importance_weights/(np.sum(unnormalized_importance_weights))
    new_samples, new_weights, indices = self.systematic_resample(samples, importance_weights)
    new_samples_moved=self.metropolis_hasting_various_steps(new_samples,nsteps_MH)
    new_temperature_par=self.find_next_beta(new_samples_moved, temperature_par) # Added temperature_par
    return iteration+1,new_samples_moved,new_temperature_par,new_weights,indices,evidence

  def iterations(self,nsteps_MH,n_iterations=100):
    iteration,samples,temperature,weights,indices,evidence=self.first_iteration(nsteps_MH)
    print(f"iteration: {iteration}, temperature: {temperature}, evidence: {evidence}")
    for i in range(n_iterations):
      iteration,samples,temperature,weights,indices,evidence=self.iteration(nsteps_MH,iteration,samples,temperature,weights,indices,evidence)
      print(f"iteration: {iteration}, temperature: {temperature}, evidence: {evidence}")
    return samples,temperature,weights,indices,evidence