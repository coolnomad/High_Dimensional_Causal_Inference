#' Compute the posterior mean win probability for a stopped 7W–3L draft run
#'
#' Given a draft run that ends at an absorbing boundary (7 wins or 3 losses),
#' this function returns the posterior mean of the underlying match win
#' probability \(p\) under a Beta(alpha, beta) prior.
#'
#' The likelihood is truncated because we only observe the run *at the boundary*,
#' not the full sequence of wins/losses. For a run ending in 7 wins, the data
#' consist of (7, l) where l < 3; for a run ending in 3 losses, the data consist
#' of (w, 3) where w < 7. The posterior for \(p\) is then:
#'
#'   p ~ Beta(alpha + w_obs, beta + l_obs)
#'
#' where w_obs and l_obs are the wins/losses *at the boundary*.
#'
#' A Beta(0.5, 0.5) (Jeffreys) prior is used by default because it is invariant
#' under reparameterization and works well for binomial problems with small n.
#'
#' @param w Integer: number of wins at stopping time (must be stop_wins or < stop_wins)
#' @param l Integer: number of losses at stopping time (must be stop_losses or < stop_losses)
#' @param stop_wins Integer: absorbing win boundary (default 7)
#' @param stop_losses Integer: absorbing loss boundary (default 3)
#' @param alpha Beta prior alpha parameter (default 0.5)
#' @param beta Beta prior beta parameter (default 0.5)
#'
#' @return Posterior mean of the underlying win probability p.
#'
#' @details
#' The function will throw an error if (w, l) does not correspond to a stopping
#' boundary, since interior (non-absorbing) states are unidentifiable under the
#' truncated likelihood.
#'
#' @examples
#' posterior_mean_p(7, 1)   # posterior mean if the run ended 7–1
#' posterior_mean_p(4, 3)   # posterior mean if the run ended 4–3
#'

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

# ------------------------------------------------------------------------------
# Simulate observed Δp for a perfect predictor and plot true vs observed
#
# This helper assumes a "perfect model" whose prediction equals the true
# deck effect Δp_true (deck_eff). It then simulates one run per deck under
# either:
#   - the Arena 7W–3L stopping rule, or
#   - a fixed number of games per deck,
# and plots:
#
#       x = Δp_true          (oracle / perfect prediction)
#       y = Δp_obs           (from the simulated run)
#
# The resulting plot shows how a *perfect* deck-effect model would look
# when evaluated on noisy, truncated outcomes.
# ------------------------------------------------------------------------------

simulate_obs_dp_once <- function(base_p,
                                 deck_eff,
                                 stop_wins   = 7L,
                                 stop_losses = 3L,
                                 fixed_games = NULL,
                                 seed        = NULL) {
  stopifnot(length(base_p) == length(deck_eff))
  if (!is.null(seed)) set.seed(seed)
  
  N <- length(base_p)
  clamp01 <- function(z, eps = 1e-6) pmin(pmax(z, eps), 1 - eps)
  p_true  <- clamp01(base_p + deck_eff)
  
  # stopping-rule simulator
  sim_stop <- function(p) {
    w <- 0L; l <- 0L
    while (w < stop_wins && l < stop_losses) {
      if (runif(1) < p) w <- w + 1L else l <- l + 1L
    }
    c(wins = w, losses = l)
  }
  
  # fixed-games simulator
  sim_fixed <- function(p, n_games) {
    wins   <- rbinom(1L, size = n_games, prob = p)
    losses <- n_games - wins
    c(wins = wins, losses = losses)
  }
  
  sim_fun <- if (is.null(fixed_games)) {
    function(p) sim_stop(p)
  } else {
    function(p) sim_fixed(p, n_games = fixed_games)
  }
  
  AB <- t(vapply(p_true, sim_fun, numeric(2)))
  wins   <- AB[, "wins"]
  losses <- AB[, "losses"]
  total  <- wins + losses
  p_mle  <- wins / pmax(total, 1L)
  
  dp_obs <- p_mle - base_p
  
  data.frame(
    base_p   = base_p,
    true_dp  = deck_eff,
    obs_dp   = dp_obs,
    wins     = wins,
    losses   = losses,
    games    = total,
    rule     = if (is.null(fixed_games)) "stop" else "fixed",
    fixed_games = if (is.null(fixed_games)) NA_integer_ else fixed_games
  )
}


plot_perfect_true_vs_obs <- function(base_p,
                                     deck_eff,
                                     stop_wins   = 7L,
                                     stop_losses = 3L,
                                     fixed_games = NULL,
                                     seed        = 123,
                                     title       = NULL) {
  df <- simulate_obs_dp_once(
    base_p      = base_p,
    deck_eff    = deck_eff,
    stop_wins   = stop_wins,
    stop_losses = stop_losses,
    fixed_games = fixed_games,
    seed        = seed
  )
  
  if (is.null(title)) {
    if (is.null(fixed_games)) {
      title <- "Perfect predictor: true Δp vs observed Δp (7–3 stop rule)"
    } else {
      title <- sprintf("Perfect predictor: true Δp vs observed Δp (%d games)", fixed_games)
    }
  }
  
  # deciles for coloring
  df$dec <- cut_number(df$base_p, 10)
  df$w   <- df$games
  
  # binned means along x = true Δp
  wm <- df %>%
    mutate(bin = ntile(true_dp, 20)) %>%
    group_by(bin) %>%
    summarise(
      x = weighted.mean(true_dp, w),
      y = weighted.mean(obs_dp,  w),
      w = sum(w),
      .groups = "drop"
    )
  
  # weighted calibration line + summary stats
  fit <- lm(obs_dp ~ true_dp, data = df)
  b   <- coef(fit)
  ann <- sprintf("intercept = %.3f, slope = %.3f\nR^2 = %.3f, RMSE = %.3f",
                 b[1], b[2],
                 summary(fit)$r.squared,
                 sqrt(weighted.mean((df$obs_dp - df$true_dp)^2, df$w)))
  
  p <- ggplot(df, aes(true_dp, obs_dp)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +  # ideal
    geom_point(aes(colour = dec, size = w), alpha = 0.25) +
    geom_smooth(aes(weight = w), method = "lm", se = FALSE, colour = "black") +
    geom_point(data = wm, aes(x = x, y = y, size = w), inherit.aes = FALSE,
               shape = 21, fill = NA, colour = "black", stroke = 0.6) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.2, label = ann) +
    labs(
      title = title,
      x     = "True deck effect Δp (perfect prediction)",
      y     = "Observed deck effect Δp (from run)",
      colour = "base_p decile"
    ) +
    theme_minimal() +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
  
  list(data = df, plot = p,
       stats = list(intercept = b[1], slope = b[2],
                    R2 = summary(fit)$r.squared,
                    RMSE = sqrt(weighted.mean((df$obs_dp - df$true_dp)^2, df$w))))
}
