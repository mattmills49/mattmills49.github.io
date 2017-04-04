library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
options(dplyr.width = Inf)

pass_col_types <- cols(
  .default = col_integer(),
  date = col_character(),
  time = col_character(),
  passer = col_character(),
  team = col_character(),
  recipient = col_character(),
  team.1 = col_character(),
  x = col_double(),
  y = col_double(),
  endX = col_double(),
  endY = col_double(),
  hteam = col_character(),
  ateam = col_character()
)

pass_df <- read_csv(file = "~/Documents/soccer.csv", col_types = pass_col_types)

bind_rows(select(pass_df, team, x, y) %>% mutate(Type = "Start"),
          select(pass_df, team, endX, endY) %>% mutate(Type = "Finish") %>% rename(x = endX, y = endY)) %>%
  mutate(xYards = x * .75, yYards = y * 1.1) %>%
  mutate(Type = factor(Type, levels = c("Start", "Finish"))) %>%
  ggplot(aes(x = xYards, y = yYards)) +
  stat_density_2d(geom = "raster", aes(fill = ..density..), contour = FALSE) +
  facet_grid(Type ~ team) +
  #coord_equal() +
  scale_fill_viridis() +
  theme(legend.position = "none")

pass_df %>%
  mutate(xYards = x * .75, yYards = y * 1.1, xEndYards =  endX * .75, yEndYards = endY * 1.1) %>%
  ggplot(aes(x = xYards, xend = xEndYards, y = yYards, yend = yEndYards)) +
  geom_segment(alpha = .2) +
  facet_wrap(~ team, nrow = 1)

# player pass

ggplot(aes(x = x, y = y), data = pass_df) +
  geom_point(alpha = .2) +
  geom_point(aes(x = x, y = y), data = filter(pass_df, passer == "David Guzman"), color = "red") 
