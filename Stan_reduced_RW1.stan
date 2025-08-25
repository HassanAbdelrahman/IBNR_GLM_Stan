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
  vector[L] alpha_raw;
  vector[D] beta_raw;
  matrix[12,D] gamma_raw;

  real<lower=0> sigma_alpha;
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

  // Smooth prior on alpha_raw (random walk + sum-to-zero)
  alpha_raw[1] ~ normal(0, sigma_alpha);
  for (t in 2:L)
    alpha_raw[t] ~ normal(alpha_raw[t-1], sigma_alpha);
  
  sigma_alpha ~ normal(0, 0.5); // half-normal (choose scale to match log-rate scale)


  // Independent prior on beta_raw (sum-to-zero will apply)
  beta_raw ~ normal(0, 1);

  to_vector(gamma_raw) ~ normal(0, 1);
}
