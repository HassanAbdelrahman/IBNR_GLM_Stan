library(dplyr)
library(lubridate)
library(tidyr)

set.seed(123)

# Simulate portfolio data
n_policy <- 1000
start_date <- as.Date("2020-01-01")
end_date <- as.Date("2022-12-31")

portfolio <- data.frame(
  ID = 1:n_policy,
  start_date = sample(seq(start_date, as.Date("2020-06-01"), by="month"), n_policy, replace = TRUE),
  end_date = sample(seq(as.Date("2022-01-01"), end_date, by="month") - 1, n_policy, replace = TRUE),
  x1 = rnorm(n_policy, mean = 50, sd = 10),
  x2 = sample(c("A","B"), n_policy, replace = TRUE)
)

# Simulate claims data with delay in days
claims <- portfolio %>%
  rowwise() %>%
  mutate(
    has_claim = rbinom(1, 1, 0.2),  # 20% chance of claim
    loss_date = if(has_claim == 1) sample(seq(start_date, end_date, by="day"), 1) else as.Date(NA),
    reporting_delay_days = if(has_claim == 1) sample(0:(5*30), 1) else NA, # delay up to ~5 months
    reporting_date = if(has_claim == 1) loss_date + reporting_delay_days else as.Date(NA)
  ) %>%
  ungroup() %>%
  filter(has_claim == 1) %>%
  dplyr::select(ID, loss_date, reporting_date)

# Define periods t = 1,2,... based on a start period
start_period <- as.Date("2020-01-01")

# period index (months since start)
date_to_period <- function(date, start) {
  interval(start, date) %/% months(1) + 1
}

claims <- claims %>%
  mutate(
    period = date_to_period(loss_date, start_period),
    delay = date_to_period(reporting_date, start_period) - date_to_period(loss_date, start_period)  # max delay = 5 months
  )


# Create run-off triangle
max_period <- max(claims$period, na.rm = TRUE)
max_delay <- max(claims$delay, na.rm = TRUE)

# initialize empty triangle
run_off_triangle <- matrix(0, nrow = max_period, ncol = max_delay + 1)

for(i in 0:max_delay){
  tbl <- table(claims$period[claims$delay == i])
  run_off_triangle[as.numeric(names(tbl)), i + 1] <- as.numeric(tbl)
}

colnames(run_off_triangle) <- paste0("d", 0:max_delay)
rownames(run_off_triangle) <- paste0("t", 1:max_period)

# Compute exposure and covariates
exposure <- sapply(1:max_period, function(t){
  current_date <- start_period %m+% months(t - 1)
  sum(portfolio$start_date <= current_date & portfolio$end_date >= current_date)
})

covariates <- t(sapply(1:max_period, function(t){
  current_date <- start_period %m+% months(t - 1)
  active <- portfolio %>% filter(start_date <= current_date & end_date >= current_date)
  c(
    x1_mean = mean(active$x1),
    x2_A_pct = mean(active$x2 == "A")
  )
}))

covariates <- as.data.frame(covariates)

# Preview objects
head(run_off_triangle)
head(covariates)
head(exposure)
