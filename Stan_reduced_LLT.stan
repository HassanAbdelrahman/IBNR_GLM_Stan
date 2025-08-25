// The input data 
data {
  int<lower=0> N_obs;
  int<lower=1> L;               // number of time periods
  int<lower=1> D;               // number of delays
  
  array[N_obs] int<lower = 0> y;
  array[N_obs] real<lower = 0> exposure;
  array[N_obs] int<lower = 0, upper = L> time;
  array[N_obs] int<lower = 0, upper = D> delay;
  array[N_obs] int<lower = 0, upper = 12> month;
}

// The parameters 
parameters {
  real mu;
  vector[L] alpha_raw;   // level
  vector[L] delta_raw;   // slope
  real<lower=0> sigma_level;
  real<lower=0> sigma_slope;
  real mu0;
  real delta0;

  vector[D] beta_raw;
  matrix[12,D] gamma_raw;
  real<lower=0> sigma;
}

// Transformed parameters
transformed parameters {
  vector[L] alpha;
  vector[D] beta;
  matrix[12,D] gamma;
  vector[N_obs] log_lambda;

  // Enforce sum-to-zero constraints
  alpha = alpha_raw - mean(alpha_raw);
  beta = beta_raw - mean(beta_raw);
  for (d in 1:D)
    gamma[,d] = gamma_raw[,d] - mean(gamma_raw[,d]);

  // Log-rate for each observation
  for(i in 1:N_obs) {
    log_lambda[i] = mu + alpha[time[i]] + beta[delay[i]+1] + gamma[month[i],delay[i]+1] + log(exposure[i]);
    }
}

// The model 
model {
  // Likelihood
  y ~ neg_binomial_2_log(log_lambda, sigma);

  // Priors
  sigma ~ exponential(0.1);
  mu ~ normal(0, 1);

  // prior on alpha_raw (local linear trend)
  alpha_raw[1] ~ normal(mu0, 1);
  delta_raw[1] ~ normal(delta0, 1);
  
  for (t in 2:L) {
    alpha_raw[t] ~ normal(alpha_raw[t-1] + delta_raw[t-1], sigma_level); // level evolves with slope
    delta_raw[t] ~ normal(delta_raw[t-1], sigma_slope);                   // slope random walk
  }
  
  mu0 ~ normal(0, 1);
  delta0 ~ normal(0, 0.1);
  sigma_level ~ normal(0, 0.5);
  sigma_slope ~ normal(0, 0.2);

  // Independent prior on beta_raw 
  beta_raw ~ normal(0, 1);

  to_vector(gamma_raw) ~ normal(0, 1);
}
