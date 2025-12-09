source('Simulated_Arena.R')
seed = as.numeric(Sys.time())
sim_train <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed, interaction_mode = "product",penalty_123 = .1)
sim_test2 <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed + 5, interaction_mode = "product", penalty_123 = .1)
sim_test3 <- simulate_5card_arena(N = 20000,skill_lower = -1,skill_upper = .8,seed = seed + 50, interaction_mode = "product", penalty_123 = .1)


d_train = sim_train$df
d_valid = sim_test2$df
d_test = sim_test3$df

#### Format Data 
library(xgboost)

make_xgb_data <- function(df) {
  K <- with(df, card1 + card2 + card3 + card4 + card5)
  frac <- df[, paste0("card",1:5)] / K
  data.frame(base_p = df$base_p, frac)
}

###########
posterior_mean_p <- function(w, l, stop_wins = 7L, stop_losses = 3L,
                             alpha = 0.5, beta = 0.5) {
  # jeffreys prior by default (alpha=beta=0.5). flat prior would be 1,1
  if (l == stop_losses) {
    a <- alpha + w
    b <- beta  + stop_losses
  } else if (w == stop_wins) {
    a <- alpha + stop_wins
    b <- beta  + l
  } else stop("row does not end at a boundary")
  a / (a + b)
}
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

print(p_test)

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

print(p_obs)

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

# simulate many observed deltas per row to estimate stop-rule noise
R <- 200
dp_obs_all <- replicate(R, {
  AB <- t(vapply(p_true, sim_stop, numeric(2)))
  p_mle <- AB[,1] / (AB[,1] + AB[,2])
  p_mle - d_test$base_p
})

# noise = observed - true
noise_mat <- sweep(dp_obs_all, 1, d_test$deck_eff, FUN = "-")
noise_var <- var(as.numeric(noise_mat))  # pooled
signal_var <- var(d_test$deck_eff)

R2_upper  <- signal_var / (signal_var + noise_var)
RMSE_floor <- sqrt(noise_var)

c(R2_upper = R2_upper, RMSE_floor = RMSE_floor)

