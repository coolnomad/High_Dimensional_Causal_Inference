source('R/01_simulate_arena.R')
source('R/helper.R')
#seed = as.numeric(Sys.time())
seed = 3000
sim_train <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed, interaction_mode = "product",penalty_123 = .1)
sim_test2 <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed + 5, interaction_mode = "product", penalty_123 = .1)
sim_test3 <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed + 50, interaction_mode = "product", penalty_123 = .1)


d_train = sim_train$df
d_valid = sim_test2$df
d_test = sim_test3$df

#### Format Simulated Data 
library(xgboost)

make_xgb_data <- function(df) {
  K <- with(df, card1 + card2 + card3 + card4 + card5)
  frac <- df[, paste0("card",1:5)] / K
  data.frame(base_p = df$base_p, frac)
}

###########
# ------------------------------------------------------------------------------
# Modeling the oracle deck bump with XGBoost
#
# Goal:
#   We want to train a model that predicts the *deck effect* Δp from observable
#   quantities in a single 7W–3L Arena-style run. In the oracle world we know:
#
#     true latent win prob: p_true = base_p + deck_eff
#     observed run outcome: wins, losses  (stopped at 7–3)
#     baseline skill: base_p
#
#   But we do *not* observe deck_eff directly. Instead, we compute an estimator:
#
#       posterior_mean_p = E[p | (wins, losses)]        # from Beta posterior
#       observed_bump    = posterior_mean_p - base_p    # Δp implied by run
#
#   This observed_bump is the target the model learns to predict.
#   In real data this is exactly what we do: isolate the deck contribution
#   by subtracting off the baseline skill estimate.
#
# Steps performed in this block:
#
#   1. Vectorize the posterior-mean function so it works on full vectors.
#   2. Compute y_train, y_valid, y_test:
#         y = observed Δp = posterior_mean_p(w, l) - base_p
#      This is our regression label.
#
#   3. Create feature matrices X_train, X_valid, X_test using make_xgb_data().
#      These contain:
#          - base_p (baseline skill)
#          - card composition fractions (card1_frac ... card5_frac)
#
#   4. Wrap the feature matrices and labels into xgb.DMatrix objects.
#
#   5. Train an XGBoost regressor to predict Δp directly from deck features.
#      Model tries to learn the mapping:
#            f(base_p, deck_fractions) → true deck effect
#      Early stopping tracks validation loss.
#
#   6. Predict on validation/test sets:
#          de_hat_valid, de_hat_test
#
#   7. Compare predicted Δp to true oracle Δp (deck_eff) using:
#          - correlation (how well ordered)
#          - RMSE (how close on average)
#
#   In this oracle environment, this block answers the question:
#       “Given perfect simulation ground truth,
#        can an ML model recover the deck effect from a single stopped run?”
#
#   These results establish the feasibility bound: if XGBoost recovers the deck
#   effect here, then the real-data pipeline (with baseline skill, features,
#   and posterior bump estimation) should work as well.
# ------------------------------------------------------------------------------

posterior_mean_p_vec <- Vectorize(posterior_mean_p)

y_train <- posterior_mean_p_vec(d_train$wins, d_train$losses) - d_train$base_p
y_valid <- posterior_mean_p_vec(d_valid$wins, d_valid$losses) - d_valid$base_p
y_test  <- posterior_mean_p_vec(d_test$wins,  d_test$losses)  - d_test$base_p

X_train <- as.matrix(make_xgb_data(d_train))
X_valid <- as.matrix(make_xgb_data(d_valid))
X_test  <- as.matrix(make_xgb_data(d_test))

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dvalid <- xgb.DMatrix(data = X_valid, label = y_valid)


watchlist <- list(train = dtrain, valid = dvalid)

params <- list(
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 1.0
)

fit_xgb <- xgb.train(
  params = params,
  data   = dtrain,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  verbose = 1
)

de_hat_valid <- predict(fit_xgb, X_valid)
de_hat_test  <- predict(fit_xgb, X_test)

c(
  cor_valid = cor(de_hat_valid, d_valid$deck_eff),
  cor_test  = cor(de_hat_test,  d_test$deck_eff),
  rmse_valid = sqrt(mean((de_hat_valid - d_valid$deck_eff)^2)),
  rmse_test  = sqrt(mean((de_hat_test  - d_test$deck_eff)^2))
)

# ------------------------------------------------------------------------------
# Visualizing model performance: predicted Δp vs true/observed Δp
#
# This section creates two calibration-style plots that evaluate how well the
# XGBoost model recovers the deck effect (Δp) in the oracle environment.
#
# Part 1: TRUE Δp vs PREDICTED Δp
# --------------------------------
# We attach the model predictions (de_hat) to the validation/test data and then:
#
#   • Bin decks by predicted Δp (20 quantile bins)
#   • Compute weighted means of:
#         - predicted Δp  (x-axis)
#         - true Δp       (y-axis; oracle deck_eff)
#
#   • Fit a weighted linear regression:
#         true Δp  ~  predicted Δp
#
# The resulting plot shows:
#   • Colored points = individual decks (colored by baseline skill decile)
#   • Hollow circles = bin means (de-noised visualization)
#   • Smooth black line = weighted best-fit
#   • Dashed y=x line = perfect prediction
#
# Interpretation:
#   This graph answers the core question:
#       “How close does the model get to the true causal deck effect Δp?”
#   In the oracle world, this is the maximum achievable performance curve.
#
#
# Part 2: OBSERVED Δp vs PREDICTED Δp
# -----------------------------------
# In real Arena-style data, we *cannot* observe true Δp. We observe only:
#
#       dp_obs = (A / (A + B)) - base_p,
#
# where:
#       A = wins in the 7W–3L run,
#       B = losses,
#   and the run is stopped at an absorbing boundary.
#
# In this block we:
#   • Simulate one 7W–3L run per deck (if A/B not present)
#   • Compute the observed Δp from the run
#   • Bin by predicted Δp (same as before)
#   • Fit the weighted regression:
#         observed Δp  ~ predicted Δp
#
# The resulting plot shows the *recoverability limit* induced by stop-rule noise.
#
#
# Why two plots?
#   • TRUE Δp vs predicted Δp
#       Measures how well the model recovers the genuine causal signal.
#
#   • OBSERVED Δp vs predicted Δp
#       Shows how much that signal survives when pushed through the
#       highly noisy 7W–3L stopping process.
#
# Combined, these figures demonstrate:
#   1. Whether the model can recover the oracle deck effect in principle.
#   2. How truncated draft outcomes distort the signal in practice.
#   3. What performance metrics (slope, R², RMSE) are even achievable
#      under real Arena-style observational constraints.
#
# These plots mirror what will later be done with real MTG data:
#   predicted Δp (from model) ≈ causal deck effect
#   observed Δp (from run)    ≈ noisy measurement of that effect
# ------------------------------------------------------------------------------

# Visualize delta p vs true p 
library(dplyr)
library(ggplot2)

# add preds
d_valid$de_hat <- de_hat_valid
d_test$de_hat  <- de_hat_test

# ensure deciles + weights exist (use 1 if you didn't simulate run-lengths)
d_test <- d_test %>%
  mutate(dec = cut_number(base_p, 10),
         w   = if ("w" %in% names(d_test)) w else 1)

# weighted bin means for the hollow circles
wm <- d_test %>%
  mutate(bin = ntile(de_hat, 20)) %>%
  group_by(bin) %>%
  summarise(x = weighted.mean(de_hat, w),
            y = weighted.mean(deck_eff, w),
            w = sum(w), .groups = "drop")

# weighted calibration line + annotation
fit <- lm(deck_eff ~ de_hat, data = d_test, weights = w)
b   <- coef(fit)
ann <- sprintf("intercept = %.3f, slope = %.3f\nR^2 = %.3f, RMSE = %.3f",
               b[1], b[2],
               summary(fit)$r.squared,
               sqrt(weighted.mean((d_test$deck_eff - d_test$de_hat)^2, d_test$w)))

p_test <- ggplot(d_test, aes(de_hat, deck_eff)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +     # perfect = y = x
  geom_point(aes(colour = dec, size = w), alpha = 0.25) +
  geom_smooth(aes(weight = w), method = "lm", se = FALSE, colour = "black") +
  geom_point(data = wm, aes(x = x, y = y, size = w), inherit.aes = FALSE,
             shape = 21, fill = NA, colour = "black", stroke = 0.6) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.2, label = ann) +
  labs(title = "TRUE Δp vs predicted Δp (TEST)",
       x = "Predicted Δp (deck effect)",
       y = "TRUE Δp (deck effect)", colour = "base_p decile") +
  theme_minimal() +
  theme(plot.background  = element_rect(fill="white", colour=NA),
        panel.background = element_rect(fill="white", colour=NA))

#print(p_test)

##### 
library(dplyr)
library(ggplot2)

set.seed(42)

# 1) if A/B not present, simulate a 7/3 arena run per row from true p = base_p + deck_eff
if (!all(c("A","B") %in% names(d_test))) {
  clamp01 <- function(z) pmin(pmax(z, 1e-6), 1-1e-6)
  p_true  <- clamp01(d_test$base_p + d_test$deck_eff)
  
  sim_stop <- function(p) {
    w <- 0L; l <- 0L
    while (w < 7L && l < 3L) {
      if (runif(1) < p) w <- w + 1L else l <- l + 1L
    }
    c(A = min(w, 7L), B = min(l, 3L))
  }
  AB <- t(vapply(p_true, sim_stop, numeric(2)))
  d_test$A <- AB[, "A"]; d_test$B <- AB[, "B"]
}

# 2) observed quantities
d_test <- d_test %>%
  mutate(w      = A + B,
         p_mle  = A / pmax(A + B, 1),
         dp_obs = p_mle - base_p,
         dec    = cut_number(base_p, 10))

# 3) binned means for hollow circles (by predicted Δp)
wm_obs <- d_test %>%
  mutate(bin = ntile(de_hat, 20)) %>%
  group_by(bin) %>%
  summarise(x = weighted.mean(de_hat, w),
            y = weighted.mean(dp_obs, w),
            w = sum(w), .groups = "drop")

# 4) weighted calibration line + annotation
fit_obs <- lm(dp_obs ~ de_hat, data = d_test, weights = w)
b <- coef(fit_obs)
ann <- sprintf("intercept = %.3f, slope = %.3f\nR^2 = %.3f, RMSE = %.3f",
               b[1], b[2],
               summary(fit_obs)$r.squared,
               sqrt(weighted.mean((d_test$dp_obs - d_test$de_hat)^2, d_test$w)))

# 5) plot
p_obs <- ggplot(d_test, aes(de_hat, dp_obs)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") + # ideal y=x
  geom_point(aes(colour = dec, size = w), alpha = 0.25) +
  geom_smooth(aes(weight = w), method = "lm", se = FALSE, colour = "black") +
  geom_point(data = wm_obs, aes(x = x, y = y, size = w), inherit.aes = FALSE,
             shape = 21, fill = NA, colour = "black", stroke = 0.6) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.2, label = ann) +
  labs(title = "OBSERVED Δp vs predicted Δp (TEST)",
       x = "Predicted Δp (deck effect)",
       y = "Observed Δp (MLE)",
       colour = "base_p decile") +
  theme_minimal() +
  theme(plot.background  = element_rect(fill="white", colour=NA),
        panel.background = element_rect(fill="white", colour=NA))

#print(p_obs)

# create figs directory if missing
if (!dir.exists("figs")) dir.create("figs")

# save the oracle TRUE Δp vs predicted Δp plot
ggsave("figs/pred_vs_true_dp.png", p_test,
       width = 7, height = 6, dpi = 300)

# save the observed Δp vs predicted Δp plot
ggsave("figs/pred_vs_obs_dp.png", p_obs,
       width = 7, height = 6, dpi = 300)


## Compute what's roughly the best R2 we can get from the truncated runs if we nailed the deck effect
# true and observed deltas
clamp01 <- function(z) pmin(pmax(z, 1e-6), 1-1e-6)
p_true  <- clamp01(d_test$base_p + d_test$deck_eff)

sim_stop <- function(p){
  w <- l <- 0L
  while(w < 7L && l < 3L){
    if (runif(1) < p) w <- w + 1L else l <- l + 1L
  }
  c(A=w, B=l)
}
# ------------------------------------------------------------------------------
# Estimate the noise floor and R² upper bound for deck-effect recovery
#
# This function quantifies how much of the true deck effect (Δp_true) can ever
# be recovered from observed Arena-style results, given a particular play rule.
#
# Two regimes:
#   1) Stopping rule (default): stop at stop_wins or stop_losses (7–3).
#   2) Fixed-games rule: play exactly fixed_games matches per deck.
#
# For each deck, we simulate R independent runs, compute the observed bump:
#
#       Δp_obs = p_hat - base_p
#
# and compare Δp_obs to Δp_true = deck_eff. The variance of the difference
# gives the irreducible noise induced by the play rule.
#
# Inputs:
#   base_p      numeric vector of baseline win probabilities (per-deck)
#   deck_eff    numeric vector of true deck effects Δp_true (per-deck)
#   R           number of replicated runs per deck (default 200)
#   stop_wins   absorbing win boundary (default 7; used if fixed_games = NULL)
#   stop_losses absorbing loss boundary (default 3; used if fixed_games = NULL)
#   fixed_games integer; if non-NULL, ignore stopping rule and play exactly
#               this many matches per deck
#
# Outputs:
#   A named list with:
#     - R2_upper   : theoretical max R² between Δp_obs and Δp_true
#     - RMSE_floor : irreducible RMSE between Δp_obs and Δp_true
#     - signal_var : Var(Δp_true)
#     - noise_var  : Var(Δp_obs - Δp_true)
#
# Example usage:
#   nf_7_3   <- estimate_noise_floor(d_test$base_p, d_test$deck_eff)
#   nf_20g   <- estimate_noise_floor(d_test$base_p, d_test$deck_eff,
#                                    fixed_games = 20)
#
#   nf_7_3$R2_upper    # ceiling under 7–3
#   nf_20g$R2_upper    # ceiling if we let each deck play 20 games
# ------------------------------------------------------------------------------

estimate_noise_floor <- function(base_p,
                                 deck_eff,
                                 R           = 200L,
                                 stop_wins   = 7L,
                                 stop_losses = 3L,
                                 fixed_games = NULL) {
  stopifnot(length(base_p) == length(deck_eff))
  N <- length(base_p)
  
  # clamp p into (0,1)
  clamp01 <- function(z, eps = 1e-6) pmin(pmax(z, eps), 1 - eps)
  p_true  <- clamp01(base_p + deck_eff)
  
  # simulate one run under stopping rule
  sim_stop <- function(p) {
    w <- 0L; l <- 0L
    while (w < stop_wins && l < stop_losses) {
      if (runif(1) < p) w <- w + 1L else l <- l + 1L
    }
    c(wins = w, losses = l)
  }
  
  # simulate one run under fixed-games rule
  sim_fixed <- function(p, n_games) {
    wins <- rbinom(1L, size = n_games, prob = p)
    losses <- n_games - wins
    c(wins = wins, losses = losses)
  }
  
  # choose simulator based on fixed_games
  sim_fun <- if (is.null(fixed_games)) {
    function(p) sim_stop(p)
  } else {
    function(p) sim_fixed(p, n_games = fixed_games)
  }
  
  # replicate R runs per deck, collect observed Δp
  dp_obs_all <- replicate(R, {
    AB <- t(vapply(p_true, sim_fun, numeric(2)))
    wins   <- AB[, "wins"]
    losses <- AB[, "losses"]
    total  <- wins + losses
    p_mle  <- wins / pmax(total, 1L)
    p_mle - base_p
  })
  
  # noise = observed - true
  noise_mat  <- sweep(dp_obs_all, 1, deck_eff, FUN = "-")
  noise_var  <- var(as.numeric(noise_mat))  # pooled across decks × runs
  signal_var <- var(deck_eff)
  
  R2_upper   <- signal_var / (signal_var + noise_var)
  RMSE_floor <- sqrt(noise_var)
  
  list(
    R2_upper   = R2_upper,
    RMSE_floor = RMSE_floor,
    signal_var = signal_var,
    noise_var  = noise_var
  )
}

nf_7_3   <- estimate_noise_floor(d_test$base_p, d_test$deck_eff )
nf_20g   <- estimate_noise_floor(d_test$base_p, d_test$deck_eff, fixed_games = 20)

nf_7_3
nf_20g


# 7–3 stop-rule case (Arena-style)
perf_7_3 <- plot_perfect_true_vs_obs(
  base_p   = d_test$base_p,
  deck_eff = d_test$deck_eff,
  stop_wins   = 7L,
  stop_losses = 3L,
  fixed_games = NULL
)

perf_7_3$stats   # see intercept, slope, R2, RMSE
perf_7_3$plot

# fixed 20-game case
perf_20g <- plot_perfect_true_vs_obs(
  base_p   = d_test$base_p,
  deck_eff = d_test$deck_eff,
  fixed_games = 20,
  seed = 42
)

perf_20g$stats
perf_20g$plot

if (!dir.exists("figs")) dir.create("figs")

ggsave("figs/perfect_true_vs_obs_7_3.png", perf_7_3$plot,
       width = 7, height = 6, dpi = 300)

ggsave("figs/perfect_true_vs_obs_20g.png", perf_20g$plot,
       width = 7, height = 6, dpi = 300)




