# ============================================================
# 01_real_data_deck_effect.R
# ------------------------------------------------------------
# Real-data deck effect estimation (Arena 7–3 runs)
#
# Target estimand (§2.3):
#   tau(d1) = E_P[ E[W | do(D=d1), P] - E[W | do(D=d0(P)), P] ]
#
# Operational pipeline:
#   - Observe wins/losses from stopped 7–3 runs (no direct access to p(D,P))
#   - Use Jeffreys posterior mean p_post = E[p | wins, losses] as proxy for p(D,P)
#   - Use user_game_win_rate_bucket (optionally shrunk by user_n_games_bucket) as p_base(P)
#   - Define bump_obs = p_post - p_base(P)
#   - Fit model bump_hat = E[bump_obs | D, P] using deck composition features
#   - Produce per-draft deck effects and core diagnostics
#
# Outputs:
#   - model bundle:  <models_dir>/deck_bump_xgb_bundle.rds
#   - predictions:  <models_dir>/deck_effect_predictions.parquet
#   - plots:        <figs_dir>/*.png
# ============================================================

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(ggplot2)
  library(xgboost)
})

# ----------------------------
# Paths
# ----------------------------
decks_path  <- "C:/GitHub/MTGA_DraftHelper/data/processed/decks.parquet"
figs_dir    <- "C:/GitHub/High_Dimensional_Causal_Inference/figs"
models_dir  <- "C:/GitHub/High_Dimensional_Causal_Inference/models"

dir.create(figs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Utilities
# ----------------------------
clip01 <- function(p, eps = 1e-6) pmin(pmax(p, eps), 1 - eps)
logit <- function(p) log(p / (1 - p))
invlogit <- function(z) 1 / (1 + exp(-z))

posterior_mean_p <- function(w, l, a0 = 0.5, b0 = 0.5) {
  # Jeffreys prior Beta(0.5, 0.5)
  (w + a0) / (w + l + a0 + b0)
}

save_plot <- function(p, filename, width = 8, height = 5, dpi = 160) {
  ggsave(filename = file.path(figs_dir, filename),
         plot = p, width = width, height = height, dpi = dpi)
}

# ----------------------------
# 0) Load data
# ----------------------------
x <- as.data.table(read_parquet(decks_path))

required_cols <- c(
  "draft_id",
  "event_match_wins", "event_match_losses", "n_games",
  "user_game_win_rate_bucket", "user_n_games_bucket"
)

missing_cols <- setdiff(required_cols, names(x))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# ----------------------------
# 1) Define deck feature columns (intervention space D)
# ----------------------------
deck_cols <- grep("^deck_", names(x), value = TRUE)
deck_cols <- setdiff(deck_cols, c("deck_size_avg"))  # non-card field

if (length(deck_cols) < 50) {
  warning("Unexpectedly few deck columns found. deck_cols length = ", length(deck_cols))
}

# ----------------------------
# 2) Outcomes under 7–3 censoring: p_post
# ----------------------------
x[, A := as.integer(event_match_wins)]
x[, B := as.integer(event_match_losses)]
x[, games := as.integer(A + B)]

# basic filters
x <- x[games > 0]
# if n_games is trustworthy, enforce it; otherwise skip this check
if (!all(is.na(x$n_games))) {
  bad <- x[!is.na(n_games) & n_games != games, .N]
  if (bad > 0) warning("Found ", bad, " rows where n_games != wins+losses; using wins+losses as games.")
}

x[, p_post := posterior_mean_p(A, B)]
x[, p_post := clip01(p_post)]

# ----------------------------
# 3) Baseline skill proxy p_base(P)
# ----------------------------
# user_game_win_rate_bucket appears to be numeric levels (e.g., 0.48, 0.52, ...)
x[, base_p_raw := as.numeric(as.character(user_game_win_rate_bucket))]
x <- x[!is.na(base_p_raw)]
x[, base_p_raw := clip01(base_p_raw)]

# ----------------------------
# 4) Reliability shrinkage of baseline skill using user_n_games_bucket
# ----------------------------
# Map history bucket -> reliability weight (monotone; edit as desired)
hist_w_map <- c(`1`=1, `5`=2, `10`=3, `50`=6, `100`=8, `500`=12, `1000`=14)

x[, hist_w := hist_w_map[as.character(user_n_games_bucket)]]
x[is.na(hist_w), hist_w := 3]

mu <- mean(x$base_p_raw)
lambda <- 5  # shrink strength (tune if desired)

x[, base_p := invlogit((hist_w * logit(base_p_raw) + lambda * logit(mu)) / (hist_w + lambda))]
x[, base_p := clip01(base_p)]

# ----------------------------
# 5) Observed bump label
# ----------------------------
x[, bump_obs := p_post - base_p]

# ----------------------------
# 6) Deck normalization: counts -> fractions
# ----------------------------
# Convert deck columns to numeric matrix
deck_mat_counts <- as.matrix(x[, ..deck_cols])
storage.mode(deck_mat_counts) <- "numeric"

# deck_size = sum of included deck_cols (main deck size proxy)
x[, deck_size := rowSums(.SD), .SDcols = deck_cols]
x <- x[deck_size > 0]

# rebuild matrix after filtering
deck_mat_counts <- as.matrix(x[, ..deck_cols])
storage.mode(deck_mat_counts) <- "numeric"

deck_mat_frac <- deck_mat_counts / x$deck_size

# ----------------------------
# 7) Model: learn E[bump_obs | D, P]
# ----------------------------
X <- cbind(base_p = x$base_p, deck_mat_frac)
y <- x$bump_obs
w <- x$games

dmat <- xgb.DMatrix(data = X, label = y, weight = w)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.5,
  min_child_weight = 10
)

set.seed(1)
cv <- xgb.cv(
  params = params,
  data = dmat,
  nrounds = 5000,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = 1
)



best_nrounds <- cv$best_iteration

fit <- xgb.train(
  params = params,
  data = dmat,
  nrounds = best_nrounds,
  verbose = 1
)

# ----------------------------
# 8) Per-draft outputs
# ----------------------------
x[, bump_hat := as.numeric(predict(fit, X))]
x[, p_hat := clip01(base_p + bump_hat)]

# ----------------------------
# 9) G-computation helper: E[W | do(D=d1)]
# ----------------------------
estimate_tau <- function(d1_frac_named, base_p_vec, fit, deck_cols) {
  # Returns E_P[ bump_hat(d1, P) ] which is the ACE relative to baseline
  stopifnot(all(deck_cols %in% names(d1_frac_named)))
  X_cf <- cbind(
    base_p = base_p_vec,
    matrix(rep(unname(d1_frac_named[deck_cols]), each = length(base_p_vec)),
           nrow = length(base_p_vec))
  )
  colnames(X_cf) <- c("base_p", deck_cols)
  mean(as.numeric(predict(fit, X_cf)))
}

# Example: tau for the *mean observed composition* (sanity check; should be near 0)
mean_comp <- colMeans(deck_mat_frac)
names(mean_comp) <- deck_cols
tau_mean_comp <- estimate_tau(mean_comp, x$base_p, fit, deck_cols)

cat("\nSanity: tau(mean observed composition) =", round(tau_mean_comp, 6), "\n")

weighted_mean <- weighted.mean(x$bump_obs, w = x$games)
cat("\nSanity: weigthed mean bump:",round(weighted_mean,6))
# ----------------------------
# 10) Diagnostics + plots
# ----------------------------
# create folds explicitly
set.seed(1)
K <- 5
fold_id <- sample(rep(1:K, length.out = nrow(x)))

oof_pred <- numeric(nrow(x))

for (k in 1:K) {
  train_idx <- fold_id != k
  test_idx  <- fold_id == k
  
  dtrain <- xgb.DMatrix(X[train_idx, ], label = y[train_idx], weight = w[train_idx])
  dtest  <- xgb.DMatrix(X[test_idx, ])
  
  fit_k <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  oof_pred[test_idx] <- predict(fit_k, dtest)
}

# CV R² (weighted)
cv_fit <- lm(y ~ oof_pred, weights = w)
summary(cv_fit)$r.squared

x[, bump_hat_oof := oof_pred]
x[, p_hat_oof := clip01(base_p + bump_hat_oof)]

# (i) Centering check
center_bump <- weighted.mean(x$bump_obs, w = x$games)
cat("Centering: weighted mean(bump_obs) =", round(center_bump, 6), "\n")

p1 <- ggplot(x, aes(x = bump_obs, weight = games)) +
  geom_histogram(bins = 60, color = "white") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(
    title = "Observed bump (p_post - base_p), weighted by games",
    x = "bump_obs",
    y = "Weighted count"
  )
save_plot(p1, "bump_obs_hist.png")

# (ii) Calibration-style plot: p_hat vs p_post
cal_bins <- 25
x[, p_hat_bin := cut(p_hat_oof, breaks = quantile(p_hat_oof, probs = seq(0, 1, length.out = cal_bins + 1)),
                     include.lowest = TRUE)]
cal <- x[, .(
  p_hat_mean  = weighted.mean(p_hat_oof, w = games),
  p_post_mean = weighted.mean(p_post, w = games),
  n = .N,
  wsum = sum(games)
), by = p_hat_bin]

p2 <- ggplot(cal, aes(x = p_hat_mean, y = p_post_mean)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Calibration: predicted p_hat vs posterior p_post",
    x = "Mean predicted p_hat (bin)",
    y = "Mean posterior p_post (bin)"
  )
save_plot(p2, "calibration_p_hat_vs_p_post.png")

# (iii) Skill leakage check: bump_hat vs base_p
p3 <- ggplot(x[sample(.N, min(.N, 20000))], aes(x = base_p, y = bump_hat_oof)) +
  geom_point(alpha = 0.25) +
  geom_smooth(method = "loess", se = FALSE) +
  labs(
    title = "Skill leakage check: bump_hat vs base_p",
    x = "base_p (shrunk baseline skill proxy)",
    y = "bump_hat"
  )
save_plot(p3, "bump_hat_vs_base_p.png")

# (iv) Observed vs predicted bump
p4 <- ggplot(x[sample(.N, min(.N, 20000))], aes(x = bump_obs, y = bump_hat_oof)) +
  geom_point(alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Bump fit: observed bump_obs vs predicted bump_hat",
    x = "bump_obs",
    y = "bump_hat"
  )
save_plot(p4, "bump_obs_vs_bump_hat.png")

# ============================================================
# Replace/upgrade diagnostic plots to match simulation style
# - Uses OUT-OF-FOLD predictions (bump_hat_oof)
# - Adds base_p deciles (colour), games (size), weighted LM fit
# - Adds binned (weighted-mean) points + annotation with R^2
# ============================================================

# --- prerequisites: x is data.table with columns bump_obs, base_p, games, bump_hat_oof ---
# If not already present:
# x[, bump_hat_oof := oof_pred]
# x[, p_hat_oof := clip01(base_p + bump_hat_oof)]

# ----------------------------
# Helper: weighted means by bin
# ----------------------------
wm_by_bin <- function(dt, xcol, ycol, wcol, bincol) {
  dt[, .(
    x = weighted.mean(get(xcol), w = get(wcol)),
    y = weighted.mean(get(ycol), w = get(wcol)),
    w = sum(get(wcol)),
    n = .N
  ), by = bincol]
}

# ----------------------------
# Create base_p deciles (like simulation dec)
# ----------------------------
# Use quantile cuts; keep as factor for discrete colour scale.
x[, base_p_dec := cut(
  base_p,
  breaks = quantile(base_p, probs = seq(0, 1, 0.1), na.rm = TRUE),
  include.lowest = TRUE
)]

# ----------------------------
# Sample for plotting (keep weights intact)
# ----------------------------
set.seed(1)
plot_dt <- x#x[sample(.N, min(.N, 20000))]

# ----------------------------
# Binned weighted-means for overlay (calibration-style points)
# Bin by predicted bump_hat_oof (same as typical calibration).
# ----------------------------
n_bins <- 25
plot_dt[, pred_bin := cut(
  bump_hat_oof,
  breaks = quantile(bump_hat_oof, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE),
  include.lowest = TRUE
)]
wm_obs <- wm_by_bin(plot_dt, xcol = "bump_hat_oof", ycol = "bump_obs", wcol = "games", bincol = "pred_bin")

# ----------------------------
# Annotation: weighted R^2 from linear fit (OOF)
# ----------------------------
cv_fit <- lm(bump_obs ~ bump_hat_oof, data = x, weights = games)
r2_oof <- summary(cv_fit)$r.squared
slope  <- unname(coef(cv_fit)[2])
inter  <- unname(coef(cv_fit)[1])
rmse <- cv$evaluation_log$test_rmse_mean[best_nrounds]

ann <- sprintf("intercept = %.3f, slope = %.3f \nOOF weighted R² = %.3f, RMSE = %.3f", inter, slope, r2_oof,rmse)

# ----------------------------
# Plot: OBSERVED Δp vs predicted Δp (OOF) in simulation format
# ----------------------------
p_bump_oof <- ggplot(plot_dt, aes(x = bump_hat_oof, y = bump_obs)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +     # ideal y=x
  geom_point(aes(colour = base_p_dec, size = games), alpha = 0.25) +
  geom_smooth(aes(weight = games), method = "lm", se = FALSE, colour = "black") +
  geom_point(
    data = wm_obs,
    aes(x = x, y = y, size = w),
    inherit.aes = FALSE,
    shape = 21, fill = NA, colour = "black", stroke = 0.6
  ) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.2, label = ann) +
  labs(
    title  = "OBSERVED Δp vs predicted Δp (OOF)",
    x      = "Predicted Δp (deck bump; out-of-fold)",
    y      = "Observed Δp (p_post − base_p)",
    colour = "base_p decile",
    size   = "games"
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

save_plot(p_bump_oof, "bump_obs_vs_bump_hat_oof_simstyle.png")


# ----------------------------
# Save model bundle + predictions
# ----------------------------
bundle <- list(
  fit = fit,
  params = params,
  best_nrounds = best_nrounds,
  deck_cols = deck_cols,
  baseline = list(
    mu = mu,
    lambda = lambda,
    hist_w_map = hist_w_map
  ),
  paths = list(
    decks_path = decks_path,
    figs_dir = figs_dir,
    models_dir = models_dir
  )
)

saveRDS(bundle, file.path(models_dir, "deck_bump_xgb_bundle.rds"))

# keep a light prediction table (not full deck matrix)
pred_tbl <- x[, .(
  draft_id,
  base_p_raw, base_p,
  user_n_games_bucket,
  A, B, games,
  p_post,
  bump_obs,
  bump_hat,
  p_hat,
  deck_size
)]

write_parquet(pred_tbl, file.path(models_dir, "deck_effect_predictions.parquet"))

cat("\nSaved:\n",
    "- ", file.path(models_dir, "deck_bump_xgb_bundle.rds"), "\n",
    "- ", file.path(models_dir, "deck_effect_predictions.parquet"), "\n",
    "- plots in: ", figs_dir, "\n", sep = "")
