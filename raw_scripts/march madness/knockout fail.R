library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
options(dplyr.width = Inf)

nyt_info <- read_csv("nyt_copy.csv", col_names = "data", col_types = list(data = col_character()))

nyt_df <- matrix(nyt_info$data, ncol = 7, byrow = T) %>%
  as.data.frame(stringsAsFactors = F) %>%
  magrittr::set_names(c("Team", "CBS", "Yahoo", "fivethirtyeight", "ESPN", "Pomery", "Sagarin")) %>%
  mutate(Seed = as.numeric(drop(str_extract_all(Team, "[0-9]+", simplify = T))),
         Team = str_trim(str_replace_all(Team, "[0-9]+", ""))) %>%
  mutate_at(2:7, funs(as.numeric(drop(str_extract_all(., "[0-9]+", simplify = T)))))

           
info_538 <- read_csv("fivethirtyeight_ncaa_forecasts.csv", col_types = cols(
  gender = col_character(),
  forecast_date = col_date(format = ""),
  playin_flag = col_integer(),
  rd1_win = col_double(),
  rd2_win = col_double(),
  rd3_win = col_double(),
  rd4_win = col_double(),
  rd5_win = col_double(),
  rd6_win = col_double(),
  rd7_win = col_double(),
  team_alive = col_integer(),
  team_id = col_integer(),
  team_name = col_character(),
  team_rating = col_double(),
  team_region = col_character(),
  team_seed = col_character()
)) %>%
  filter(gender == "mens")

matchups <- info_538 %>%
  select(team_name, team_rating, team_region, team_seed, rd2_win) %>%
  mutate(team_seed = as.numeric(str_replace_all(team_seed, "[a-z]+", "")),
         opponent_seed = 17 - team_seed)

matchup_info <- left_join(matchups, select(matchups, -rd2_win), by = c("team_region", "team_seed" = "opponent_seed"))

matchup_info %>%
  ggplot(aes(x = team_rating.x, y = team_rating.y, color = rd2_win)) +
  geom_point() +
  viridis::scale_color_viridis()

library(mgcv)
matchup_model <- gam(rd2_win ~ s(team_rating.x) + s(team_rating.y), data = matchup_info)
matchup_model <- glm(rd2_win ~ team_rating.x + team_rating.y, data = matchup_info)

all_teams <- distinct(info_538, team_name, team_rating)

game_pred_tbl <- crossing(all_teams, all_teams) %>% 
  magrittr::set_names(c("team_name.x", "team_rating.x", "team_name.y", "team_rating.y"))

game_pred_tbl$prediction <- predict(matchup_model, newdata = game_pred_tbl)

sim_knockout <- function(prob_538){
  
}