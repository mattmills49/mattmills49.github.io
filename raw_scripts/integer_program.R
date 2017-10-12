library(ompr)
library(ompr.roi)
library(dplyr)
library(ROI)
library(ROI.plugin.glpk)

team_wins <- read.csv(file = "~/Downloads/fivethirtyeight_wins.csv", header = T, stringsAsFactors = F)
names(team_wins) <- c("Elo_Rating", "Team", "Wins", "Losses", "Playoff_perc", "Division_perc", "SuperBowl_perc", "Points_2016", "Proj_Points_538")

cost <- team_wins$Points_2016
proj_points <- team_wins$Proj_Points_538

five38_model <- MIPModel() %>%
  add_variable(x[i], i = 1:32, type = "binary") %>%
  add_constraint(sum_expr(x[i] * cost[i], i = 1:32) <= 50) %>%
  add_constraint(sum_expr(x[i] * cost[i], i = 1:32) >= 48) %>%
  set_objective(sum_expr(x[i] * proj_points[i], i = 1:32), "max") %>%
  solve_model(with_ROI(solver = "glpk"))

base_model <- MIPModel() %>%
  add_variable(x[i], i = 1:32, type = "binary") %>%
  add_constraint(sum_expr(x[i] * cost[i], i = 1:32) <= 50) %>%
  add_constraint(sum_expr(x[i] * cost[i], i = 1:32) >= 48) %>%
  set_objective(sum_expr(x[i] * cost[i], i = 1:32), "max") %>%
  solve_model(with_ROI(solver = "glpk"))

solution_five38 <- get_solution(five38_model, x[i])
solution_base <- get_solution(base_model, x[i])
