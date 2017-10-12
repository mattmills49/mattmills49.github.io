library(dplyr)
library(purrr)
library(ggplot2)
library(broom)
library(stringr)
library(lme4)

vec_to_df <- function(x, .names){
  setNames(data.frame(x1 = names(x), x2 = unname(x)), .names)
}

num_groups <- 50
group_counts <- round(10^runif(n = num_groups, min = .7, max = 3))
group_means <- rnorm(n = num_groups, sd = 3)

ggplot(aes(x = n, y = mean), data = data_frame(n = group_counts, mean = group_means)) +
  geom_point()

group_df <- map2(group_counts, group_means, ~ data_frame(group_mean = .y, group_value = rnorm(n = .x, mean = .y))) %>%
  bind_rows(.id = "group_id") %>%
  mutate(x1 = rnorm(n = n(), sd = 3),
         y = x1 + group_value + rnorm(n = n()))

### Standard GLM

group_glm <- glm(y ~ group_id + x1 - 1, data = group_df)

group_coefs <- tidy(group_glm) %>%
  filter(str_detect(term, "group_id")) %>%
  mutate(group_id = str_extract(term, "[0-9]+"))

group_info <- data_frame(group_id = as.character(1:50), group_obs = group_counts, group_value = group_means) %>%
  left_join(group_coefs, by = c("group_id"))

ggplot(aes(x = group_obs, y = group_value - estimate), data = group_info) +
  geom_point()

### Mixed Model

group_lmer <- lmer(y ~ (1 | group_id) + x1 - 1, data = group_df)

lmer_df <- ranef(group_lmer)[[1]]
lmer_df$group_id <- row.names(lmer_df)
names(lmer_df)[1] <- "estimate"

group_info %>%
  left_join(lmer_df, by = c("group_id")) %>% View
  ggplot(aes(x = estimate.x, y = estimate.y)) +
  geom_point()
