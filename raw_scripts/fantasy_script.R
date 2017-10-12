library(ompr)
library(ompr.roi)
library(dplyr)
library(ROI)
library(ROI.plugin.glpk)

#' Ok we have three posts a week, 14 people, and we don't want a person to have posts in the same week, and everyone needs to do at least 1 post

n_players <- 14
n_weeks <- 15
n_posts <- 3

weights <- array(runif(n_players * n_weeks * n_posts), dim = c(n_weeks, n_posts, n_players))

schedule_model <- MIPModel() %>%
  ## Week_post_player
  add_variable(x[i, j, k], i = seq_len(n_weeks), j = seq_len(n_posts), k = seq_len(n_players), type = "binary") %>%
  set_objective(sum_expr(weights[i, j, k] * x[i, j, k], i = seq_len(n_weeks), j = seq_len(n_posts), k = seq_len(n_players))) %>%
  ## each player has 3 or 4 posts
  add_constraint(sum_expr(x[i, j, k], i = seq_len(n_weeks), j = seq_len(n_posts)) <= 4, k = seq_len(n_players)) %>%
  add_constraint(sum_expr(x[i, j, k], i = seq_len(n_weeks), j = seq_len(n_posts)) >= 3, k = seq_len(n_players)) %>%
  ## each week 3 posts
  add_constraint(sum_expr(x[i, j, k], j = seq_len(n_posts), k = seq_len(n_players)) <= 3, i = seq_len(n_weeks)) %>%
  ## only one person per post
  add_constraint(sum_expr(x[i, j, k], k = seq_len(n_players)) == 1, j = seq_len(n_posts), i = seq_len(n_weeks)) %>%
  ## no player has multiple posts in a given week
  add_constraint(sum_expr(x[i, j, k], j = seq_len(n_posts)) <= 1, k = seq_len(n_players), i = seq_len(n_weeks)) %>%
  ## each player gets at least 1 of each post type
  add_constraint(sum_expr(x[i, j, k], i = seq_len(n_weeks)) >= 1, k = seq_len(n_players), j = seq_len(n_posts))
  ## No person has posts in consecutive weeks
  ##add_constraint(sum_expr(x[i, j, k], j = seq_len(n_posts))k = seq_len(n_players), i)

solved_schedule <- solve_model(schedule_model, with_ROI(solver = "glpk"))
  
solutions <- get_solution(solved_schedule, x[i, j, k])

players <- c("Matt", "Chas", "JMo", "Kyle", "Sean", "Daniel", "Luke", "Nick", "Scott", "Ryan", "Clay", "Ben", "Trey", "Will")
posts <- c("Recap", "Misc", "Preview")

post_schedule <- solutions %>%
  filter(value == 1) %>%
  mutate(author = players[k],
         posts = posts[j],
         week = i + 1) %>%
  select(week, author, posts) %>%
  tidyr::spread(posts, author)
  