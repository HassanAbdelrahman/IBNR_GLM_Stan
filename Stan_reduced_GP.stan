// GP model (non-centered alpha, Cholesky GP prior)
data {
  int<lower=0> N_obs;
  int<lower=1> L;               // number of time periods
  int<lower=1> D;               // number of delays

  array[N_obs] int<lower = 0> y;
  array[N_obs] real<lower = 0> exposure;
  array[N_obs] int<lower = 1, upper = L> time;
  array[N_obs] int<lower = 0, upper = D> delay;   // keep 0-based delays
  array[N_obs] int<lower = 1, upper = 12> month;
}

parameters {
  real mu;

  // non-centered GP parameters
  vector[L] alpha_std;           // standard normal
  real log_rho;                  // unconstrained; rho = exp(log_rho)
  real<lower=0> sigma_alpha;     // GP marginal sd (amplitude)

  vector[D] beta_raw;
  matrix[12, D] gamma_raw;

  real<lower=0> sigma;           // NegBin overdispersion (phi)
}

transformed parameters {
  real rho = exp(log_rho);
  matrix[L, L] K;
  matrix[L, L] L_K;
  vector[L] alpha;               // GP draw (non-centered -> centered below)
  vector[L] alpha_centered;
  vector[D] beta;
  matrix[12, D] gamma;
  vector[N_obs] log_lambda;

  // Build covariance matrix K (squared exponential)
  {
  vector[L] idx;
  for (n in 1:L) idx[n] = n;   // idx = 1,2,...,L as reals

  for (i in 1:L) {
    for (j in 1:L) {
      real dist = (idx[i] - idx[j]) / rho;
      K[i,j] = square(sigma_alpha) * exp(-0.5 * dist * dist);
    }
  }
    // jitter for numerical stability
    for (i in 1:L) K[i,i] += 1e-8;
  }

  // Cholesky of K
  L_K = cholesky_decompose(K);

  // non-centered transform
  alpha = L_K * alpha_std;

  // center alpha (sum-to-zero)
  alpha_centered = alpha - mean(alpha);

  // center beta and gamma
  beta = beta_raw - mean(beta_raw);
  for (d in 1:D)
    gamma[, d] = gamma_raw[, d] - mean(gamma_raw[, d]);

  // log rate
  for (i in 1:N_obs) {
    log_lambda[i] = mu
                    + alpha_centered[ time[i] ]
                    + beta[ delay[i] + 1 ] 
                    + gamma[ month[i], delay[i] + 1 ]
                    + log(exposure[i]);
  }
}

model {
  // Priors
  mu ~ normal(0, 2);
  log_rho ~ normal(0, 1.5);          // weakly informative
  sigma_alpha ~ normal(0, 1) T[0,];  // half-normal
  to_vector(beta_raw) ~ normal(0, 1);
  to_vector(gamma_raw) ~ normal(0, 1);

  // alpha_std is standard normal
  alpha_std ~ normal(0, 1);

  sigma ~ exponential(1);

  // Likelihood
  y ~ neg_binomial_2_log(log_lambda, sigma);
}
