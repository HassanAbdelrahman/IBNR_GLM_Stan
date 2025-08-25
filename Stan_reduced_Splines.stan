// The input data 
data {
  int<lower=0> N_obs;
  int<lower=1> L;               // number of time periods
  int<lower=1> D;               // number of delays
  int<lower=1> K;               // number of spline basis functions
  
  array[N_obs] int<lower = 0> y;
  array[N_obs] real<lower = 0> exposure;
  array[N_obs] int<lower = 0, upper = L> time;
  array[N_obs] int<lower = 0, upper = D> delay;
  array[N_obs] int<lower = 0, upper = 12> month;
  
  // Spline design matrix (L x K), precomputed in R
  matrix[L, K] B_spline; 
}

// The parameters 
parameters {
  real mu;
  vector[K] theta;              // spline coefficients
  vector[D] beta_raw;
  matrix[12,D] gamma_raw;

  real<lower=0> sigma_theta;    // penalty scale for splines
  real<lower=0> sigma;
}

// Transformed parameters
transformed parameters {
  vector[L] alpha;
  vector[D] beta;
  matrix[12,D] gamma;
  vector[N_obs] log_lambda;

  // Penalized spline representation
  alpha = B_spline * theta; 
  alpha = alpha - mean(alpha);  // sum-to-zero centering

  beta = beta_raw - mean(beta_raw);
  for (d in 1:D)
    gamma[,d] = gamma_raw[,d] - mean(gamma_raw[,d]);

  // Log-rate for each observation
  for(i in 1:N_obs) {
    log_lambda[i] = mu + alpha[time[i]] + beta[delay[i]+1] 
                    + gamma[month[i],delay[i]+1] 
                    + log(exposure[i]);
  }
}

// The model 
model {
  // Likelihood
  y ~ neg_binomial_2_log(log_lambda, sigma);

  // Priors
  sigma ~ exponential(0.1);
  mu ~ normal(0, 1);

  // Penalized spline prior: difference penalty on coefficients
  for (k in 3:K)
    theta[k] ~ normal(2*theta[k-1] - theta[k-2], sigma_theta);
  theta[1:2] ~ normal(0, 1);    
  sigma_theta ~ normal(0, 0.1);

  // Other effects
  beta_raw ~ normal(0, 1);
  to_vector(gamma_raw) ~ normal(0, 1);
}
