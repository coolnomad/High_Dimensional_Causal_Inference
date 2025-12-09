# Simulator function
simulate_5card_arena <- function(
    N = 10000,
    stop_wins = 7L,
    stop_losses = 3L,
    deck_min = 3L,
    deck_max = 10L,
    # skill ~ logistic by default (so base_p = invlogit(skill))
    skill_dist = c("logistic", "normal"),
    skill_loc = 0, skill_scale = 1,
    # NEW: bounds for skill
    skill_lower = -Inf,
    skill_upper =  Inf,
    skill_bound_mode = c("truncate", "clip"),  # truncate = resample, clip = cap
    # card selection logits:  logit Pr(card j | skill s) ∝ a_j + b_j * s
    sel_intercept = c(-0.3,  0.0,  0.1, -0.2, -0.1),
    sel_slope     = c(-0.2,  0.1,  0.2,  0.0,  0.3),
    # base per-card effects and the card5×skill coupling
    w      = c(-0.15, 0.13, 0.14, -0.12, 0.00),
    gamma5 = 0.10,
    # NEW: interaction effects
    # bonuses (positive) & penalties (positive magnitudes; penalties will be subtracted)
    bonus_12      = 0.030,   # card1 & card2 co-occur
    bonus_34      = 0.020,   # card3 & card4 co-occur
    penalty_123   = 0.040,   # cards 1 & 2 & 3 co-occur
    penalty_134   = 0.040,   # cards 1 & 3 & 4 co-occur
    penalty_all5  = 0.080,   # all 5 cards co-occur (big penalty)
    # how to measure “co-occur” from composition:
    #   "indicator"  -> 1 if all listed cards appear at least once, else 0
    #   "minfrac"    -> min of the listed cards' fractions
    #   "product"    -> product of the listed cards' fractions
    interaction_mode = c("minfrac", "indicator", "product"),
    deck_effect_cap = 0.15,
    seed = 42
) {
  set.seed(seed)
  skill_dist <- match.arg(skill_dist)
  skill_bound_mode <- match.arg(skill_bound_mode)
  interaction_mode <- match.arg(interaction_mode)
  
  inv_logit <- function(x) 1/(1 + exp(-x))
  clip01    <- function(p, eps = 1e-9) pmin(pmax(p, eps), 1 - eps)
  
  # --- draw skill with optional bounds ---
  draw_skill <- function(N) {
    if (skill_dist == "logistic") {
      rfun <- function(n) rlogis(n, location = skill_loc, scale = skill_scale)
    } else {
      rfun <- function(n) rnorm(n, mean = skill_loc, sd = skill_scale)
    }
    s <- rfun(N)
    
    if (!is.finite(skill_lower) && !is.finite(skill_upper)) {
      return(s)
    }
    
    if (skill_bound_mode == "clip") {
      return(pmin(pmax(s, skill_lower), skill_upper))
    }
    
    # truncate: rejection sampling until all are within [lower, upper]
    inside <- (s >= skill_lower) & (s <= skill_upper)
    while (any(!inside)) {
      n_bad <- sum(!inside)
      s[!inside] <- rfun(n_bad)
      inside <- (s >= skill_lower) & (s <= skill_upper)
    }
    s
  }
  
  # --- skill and baseline p
  skill  <- draw_skill(N)
  base_p <- inv_logit(skill)
  
  # --- deck sizes
  K <- sample(seq.int(deck_min, deck_max), N, replace = TRUE)
  
  # --- softmax helper (stable) — per-vector
  softmax <- function(z) {
    zm <- max(z)
    ez <- exp(z - zm)
    ez / sum(ez)
  }
  
  # --- draw decks: categorical draws with prob depending on skill
  counts <- matrix(0L, nrow = N, ncol = 5)
  for (i in seq_len(N)) {
    logits <- sel_intercept + sel_slope * skill[i]
    probs  <- softmax(logits)
    picks  <- sample.int(5, size = K[i], replace = TRUE, prob = probs)
    counts[i, ] <- tabulate(picks, nbins = 5)
  }
  colnames(counts) <- paste0("card", 1:5)
  
  frac <- counts / K  # composition fractions per player
  
  # --- interaction feature builder on fractions
  co_fun <- switch(
    interaction_mode,
    indicator = function(fr, idx) as.numeric(all(fr[idx] > 0)),
    minfrac   = function(fr, idx) if (all(fr[idx] > 0)) min(fr[idx]) else 0,
    product   = function(fr, idx) prod(fr[idx])
  )
  
  # compute interaction intensities for each player
  f12     <- apply(frac, 1, function(fr) co_fun(fr, c(1,2)))
  f34     <- apply(frac, 1, function(fr) co_fun(fr, c(3,4)))
  f123    <- apply(frac, 1, function(fr) co_fun(fr, c(1,2,3)))
  f134    <- apply(frac, 1, function(fr) co_fun(fr, c(1,3,4)))
  f12345  <- apply(frac, 1, function(fr) co_fun(fr, 1:5))
  
  # --- deck effect: linear in fractions + card5×skill + interactions
  deck_raw <-
    as.numeric(frac %*% w) +
    gamma5 * frac[, 5] * tanh(skill) +
    bonus_12  * f12 +
    bonus_34  * f34 -
    penalty_123  * f123 -
    penalty_134  * f134 -
    penalty_all5 * f12345
  
  deck_eff <- pmax(pmin(deck_raw,  deck_effect_cap), -deck_effect_cap)
  
  # --- final underlying p with deck effect (clipped to (0,1))
  p <- clip01(base_p + deck_eff)
  
  # --- exact E[r|p] and Var[r|p] under stopping rule
  expected_realized_wr <- function(p) {
    sapply(p, function(pi) {
      rr_w <- stop_wins / (stop_wins + 0:(stop_losses - 1))
      pr_w <- choose((stop_wins - 1) + 0:(stop_losses - 1), 0:(stop_losses - 1)) *
        (pi^stop_wins) * ((1 - pi)^(0:(stop_losses - 1)))
      rr_l <- (0:(stop_wins - 1)) / ((0:(stop_wins - 1)) + stop_losses)
      pr_l <- choose((0:(stop_wins - 1)) + (stop_losses - 1), 0:(stop_wins - 1)) *
        (pi^(0:(stop_wins - 1))) * ((1 - pi)^stop_losses)
      sum(pr_w * rr_w) + sum(pr_l * rr_l)
    })
  }
  var_realized_wr <- function(p) {
    sapply(p, function(pi) {
      rr_w <- stop_wins / (stop_wins + 0:(stop_losses - 1))
      pr_w <- choose((stop_wins - 1) + 0:(stop_losses - 1), 0:(stop_losses - 1)) *
        (pi^stop_wins) * ((1 - pi)^(0:(stop_losses - 1)))
      rr_l <- (0:(stop_wins - 1)) / ((0:(stop_wins - 1)) + stop_losses)
      pr_l <- choose((0:(stop_wins - 1)) + (stop_losses - 1), 0:(stop_wins - 1)) *
        (pi^(0:(stop_wins - 1))) * ((1 - pi)^stop_losses)
      mu <- sum(pr_w * rr_w) + sum(pr_l * rr_l)
      e2 <- sum(pr_w * rr_w^2) + sum(pr_l * rr_l^2)
      e2 - mu^2
    })
  }
  
  pred_realized <- expected_realized_wr(p)
  theo_var      <- var_realized_wr(p)
  
  # --- simulate 7W/3L run per player
  simulate_run <- function(prob) {
    w <- 0L; l <- 0L
    while (w < stop_wins && l < stop_losses) {
      if (runif(1) < prob) w <- w + 1L else l <- l + 1L
    }
    c(wins = w, losses = l)
  }
  wl <- t(vapply(p, simulate_run, FUN.VALUE = c(wins = 0L, losses = 0L)))
  wins <- wl[, "wins"]; losses <- wl[, "losses"]; games <- wins + losses
  obs_wr <- wins / games
  
  # --- output
  df <- data.frame(
    skill      = skill,
    base_p     = base_p,
    deck_eff   = deck_eff,
    p          = p,
    wins       = wins,
    losses     = losses,
    games      = games,
    obs_wr     = obs_wr,
    pred_realized = pred_realized,
    theo_var      = theo_var,
    counts,
    # include interaction intensities for inspection if you want:
    inter_12   = f12,
    inter_34   = f34,
    inter_123  = f123,
    inter_134  = f134,
    inter_12345= f12345,
    check.names = FALSE
  )
  
  meta <- list(
    N = N,
    stop_wins = stop_wins,
    stop_losses = stop_losses,
    deck_min = deck_min,
    deck_max = deck_max,
    sel_intercept = sel_intercept,
    sel_slope = sel_slope,
    w = w,
    gamma5 = gamma5,
    # interactions
    bonus_12 = bonus_12,
    bonus_34 = bonus_34,
    penalty_123 = penalty_123,
    penalty_134 = penalty_134,
    penalty_all5 = penalty_all5,
    interaction_mode = interaction_mode,
    deck_effect_cap = deck_effect_cap,
    skill_dist = skill_dist,
    skill_loc = skill_loc,
    skill_scale = skill_scale,
    skill_lower = skill_lower,
    skill_upper = skill_upper,
    skill_bound_mode = skill_bound_mode
  )
  
  list(df = df, meta = meta)
}

