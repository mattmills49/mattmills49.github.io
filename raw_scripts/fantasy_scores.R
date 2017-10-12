library(googlesheets)
library(dplyr)
library(rstanarm)
library(tidyr)
library(stringr)

my_sheets <- gs_ls()
fantasy_stats <- gs_title("Fantasy Stats")
scores <- fantasy_stats %>%
  gs_read(ws = 2)

ranks <- scores %>% 
  group_by(Week) %>%
  arrange(Week, desc(Score)) %>%
  mutate(rank = 1:n()) %>%
  ungroup

scores %>%
  select(Player, Week, Score) %>%
  group_by(Player) %>%
  mutate(sd = sd(Score)) %>%
  spread(Week, Score, sep = "_") %>%
  arrange(sd) %>%
  View


ranks %>%
  select(Player, Week, rank) %>%
  group_by(Player) %>%
  mutate(avg = mean(rank)) %>%
  spread(Week, rank, sep = "_") %>%
  arrange(avg) %>%
  select(-avg) %>%
  View

test_model <- stan_glm(Score ~ Player - 1, 
                       data = scores,
                       prior = normal(location = 80, scale = 10))

tibble::enframe(test_model$coefficients)

test_mixed <- stan_glmer(Score ~ (1 | Player) - 1,
                         data = scores,
                         prior = normal(location = 80, scale = 15))

team_coefs <- test_mixed$coefficients
team_coefs <- team_coefs + team_coefs[1]
names(team_coefs) <- str_replace_all(names(team_coefs), fixed("b[(Intercept) Player:"), "") %>% str_replace_all(fixed("]"), "")
team_coefs <- team_coefs[2:15]

tibble::enframe(team_coefs)
