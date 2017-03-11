library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(mgcv)

options(dplyr.width = Inf)
shots <- read_csv(file = "~/Documents/golf/hackathon-data-horsey.csv", col_types = cols(
  round_date = col_date(format = ""),
  round_no = col_integer(),
  hole_no = col_integer(),
  shot_no = col_integer(),
  putt = col_logical(),
  completed = col_logical(),
  left_to_pin_before = col_integer(),
  left_to_pin_after = col_integer(),
  penalty = col_logical(),
  player_name = col_character(),
  name = col_character(),
  course_id = col_integer(),
  start_tee = col_integer(),
  score = col_integer(),
  par = col_integer(),
  yards = col_integer(),
  lie_after = col_character(),
  lie_before = col_character(),
  club = col_character(),
  direction = col_character()
))

ggplot(aes(x = left_to_pin_before / 36, y = club), data = shots) + 
  geom_jitter(aes(color = club), height = .3)

#' What decisions does the player make? 
#' 
#' * Where to hit the ball? By proxy this is probably what club does he use?
#' 
#' What goes into that decision?
#' 
#' * Lie
#' * Distance
#' 
#' Could we find optimal club choice depending on Lie and Distance? How would we
#' measure that? Obviously the goal is to get a lower score. So maybe we could
#' develop that. Given your location, lie, par, original distance what is your
#' expected score. And then which club given the lie would improve that score
#' the most? oooohhhh yea I like that.
#' 
#' We would probably need more data, and might have to incorporate the extra 
#' ryder cup data first. 
#+

shots <- shots %>%
  mutate(shots_remaining = score - shot_no + 1)

ggplot(aes(x = shots_remaining), data = shots) +
  geom_histogram(binwidth = 1)

shots %>%
  select(shots_remaining) %>%
  mutate(poisson = rpois(n = n(), lambda = mean(shots_remaining)),
         normal = rnorm(n = n(), mean = mean(shots_remaining), sd = sd(shots_remaining)),
         lognormal = rlnorm(n = n(), meanlog = mean(log(shots_remaining)), sdlog = sd(log(shots_remaining)))) %>%
  gather(Distribution, Values) %>%
  ggplot(aes(x = Values)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~ Distribution, nrow = 2)

ggplot(aes(x = left_to_pin_before / 36, y = shots_remaining), data = shots) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3, width = 0) +
  scale_color_discrete(guide = F)

ggplot(aes(x = par, y = shots_remaining), data = shots_with_scores) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3, width = .3) +
  scale_color_discrete(guide = F)

ggplot(aes(x = yards, y = shots_remaining), data = shots_with_scores) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3, width = 0) +
  scale_color_discrete(guide = F)

ggplot(aes(x = lie_before, y = shots_remaining), data = shots_with_scores) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3) +
  scale_color_discrete(guide = F)

ggplot(aes(x = left_to_pin_before / 36, y = shots_remaining, color = lie_before), data = shots) +
  geom_jitter(height = .3, width = 0, alpha = .3)

shots$lie_before_factor <- as.factor(shots$lie_before) ## GAM needs by the by variable to be a factor
initial_model <- gam(shots_remaining ~ s(left_to_pin_before, by = lie_before_factor),
                     data = shots,
                     family = "poisson")

data_grid <- crossing(lie_before_factor = shots$lie_before_factor, left_to_pin_before = seq(1, 500) * 36)
data_grid$expected_shots_remaining <- as.numeric(unname(predict(initial_model, data_grid, type = "response")))

ggplot(aes(x = left_to_pin_before / 36, y = expected_shots_remaining, color = lie_before_factor), data = data_grid) + 
  geom_point(alpha = .4) +
  coord_cartesian(xlim = c(0, 500), ylim = c(0, 7)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))

lie_range <- shots %>%
  group_by(lie_before_factor) %>%
  summarize(p_lower = quantile(left_to_pin_before, probs = .05), p_upper = quantile(left_to_pin_before, probs = .95))

data_grid %>%
  left_join(lie_range, by = c("lie_before_factor")) %>% 
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower) %>%
  ggplot(aes(x = left_to_pin_before / 36, y = expected_shots_remaining, color = lie_before_factor)) +
  geom_point(alpha = .4) +
  coord_cartesian(xlim = c(0, 500), ylim = c(0, 7)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))

plot_df <- data_grid %>%
  left_join(lie_range, by = c("lie_before_factor")) %>% 
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower)

ggplot() + 
  geom_point(aes(x = left_to_pin_before / 36, y = expected_shots_remaining), color = "grey", data = select(plot_df, -lie_before_factor)) +
  geom_point(aes(x = left_to_pin_before / 36, y = expected_shots_remaining, color = lie_before_factor), alpha = .4, data = plot_df) +
  facet_wrap(~lie_before_factor) +
  coord_cartesian(xlim = c(0, 500), ylim = c(0, 7)) +
  scale_color_discrete(guide = F) +
  theme_bw()

# ordered cat
cats <- sort(unique(shots$shots_remaining))
ordered_model <- gam(shots_remaining ~ s(left_to_pin_before, by = lie_before_factor),
                     data = shots,
                     family = ocat(theta = cats[c(-1, -8)]))

ordered_predictions <- predict.gam(ordered_model, newdata = data_grid, type = "response")
preds <- ordered_predictions %*% matrix(cats, ncol = 1) %>% drop %>% unname
data_grid$ordered_shots_remaining <- preds

ordered_predictions %>%
  as.data.frame %>%
  magrittr::set_names(paste0("prob_", cats)) %>%
  select(prob_1:prob_6) %>%
  cbind(data_grid) %>%
  left_join(lie_range, by = c("lie_before_factor")) %>% 
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower) %>%
  filter(lie_before_factor %in% c("Fairway", "Rough", "Tee", "Green")) %>%
  gather(key = Score, value = Probability, contains("prob_")) %>% 
  mutate(Score = stringr::str_replace_all(Score, "prob_", "")) %>%
  ggplot(aes(x = left_to_pin_before / 36, y = Probability, group = Score)) +
  geom_line(size = 2) +
  geom_line(aes(color = Score), size = 1.5) +
  scale_color_brewer() +
  #viridis::scale_color_viridis(discrete = T) +
  facet_wrap(~lie_before_factor, nrow = 2, scales = "free_x") +
  theme_minimal() +
  scale_y_continuous(labels = function(x) paste0(x * 100, "%")) +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold")) 

data_grid %>%
  left_join(lie_range, by = c("lie_before_factor")) %>% 
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower) %>%
  ggplot(aes(x = left_to_pin_before / 36, y = ordered_shots_remaining, color = lie_before_factor)) +
  geom_point(alpha = .4) +
  coord_cartesian(xlim = c(0, 500), ylim = c(0, 7)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))

ggplot() + 
  geom_point(aes(x = left_to_pin_before / 36, y = ordered_shots_remaining), color = "grey", data = select(plot_df, -lie_before_factor)) +
  geom_point(aes(x = left_to_pin_before / 36, y = ordered_shots_remaining, color = lie_before_factor), alpha = .4, data = plot_df) +
  facet_wrap(~lie_before_factor) +
  coord_cartesian(xlim = c(0, 500), ylim = c(0, 7)) +
  scale_color_discrete(guide = F) +
  theme_bw()

shots$predicted_shots_poisson <- as.numeric(unname(predict(initial_model, shots, type = "response")))

predicted_shots_ordinal <- predict.gam(ordered_model, newdata = shots, type = "response")
shots$predicted_shots_ordinal <- predicted_shots_ordinal %*% matrix(cats, ncol = 1) %>% drop %>% unname

summarize(shots, cor = cor(predicted_shots_poisson, predicted_shots_ordinal), poisson_mse = mean((shots_remaining - predicted_shots_poisson)^2), ordinal_mse = mean((shots_remaining - predicted_shots_ordinal)^2))

shot_value <- shots %>%
  group_by(course_id, round_no, hole_no) %>%
  arrange(course_id, round_no, hole_no, shot_no) %>%
  mutate(after_shot_esr = lead(predicted_shots_ordinal)) %>%
  ungroup %>%
  mutate(after_shot_esr = ifelse(is.na(after_shot_esr), 0, after_shot_esr)) %>%
  mutate(shot_value = predicted_shots_ordinal - after_shot_esr)

shots %>%
  group_by(club) %>%
  arrange(club, left_to_pin_before) %>%
  mutate(club_usage = 1, club_pct = cumsum(club_usage) / n()) %>%
  ggplot(aes(x = left_to_pin_before, y = club_pct, color = club)) + 
  geom_line()

shots %>%
  ggplot(aes(x = left_to_pin_before, fill = club)) + 
  geom_density(alpha = .6) +
  coord_cartesian(ylim = c(0, .0005))

shot_value$club_factor <- factor(shot_value$club)

club_gam <- gam(shot_value ~ s(left_to_pin_before, by = club_factor) + s(left_to_pin_before, by = lie_before_factor),
                data = mutate(shot_value, lie_before_factor = factor(lie_before)))

club_grid <- shot_value %>%
  expand(nesting(left_to_pin_before, predicted_shots_ordinal), club_factor = unique(club_factor), lie_before_factor = factor(unique(lie_before)))

club_grid$predicted_value <- as.numeric(unname(predict(club_gam, newdata = club_grid, type = "response")))

club_range <- shot_value %>%
  group_by(club_factor) %>%
  summarize(p_lower = quantile(left_to_pin_before, probs = .05), p_upper = quantile(left_to_pin_before, probs = .95))

club_grid %>%
  left_join(club_range, by = c("club_factor")) %>%
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower) %>%
  ggplot(aes(x = left_to_pin_before / 36, y = predicted_value, color = club_factor)) +
  geom_point() +
  facet_wrap(~lie_before_factor, nrow = 3, scales = "free")