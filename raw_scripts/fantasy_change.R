library(rvest)
library(dplyr)
library(stringr)
library(purrr)

positions <- c(0, 2, 4, 6)
indexes <- c(0, 50)
url <- "http://games.espn.com/ffl/leaders?slotCategoryId=%i&seasonId=2016&startIndex=%i"

position_vec <- c("QB", "RB", "WR", "TE")

fantasy_list <- list()
for(p in seq_along(positions)){
  for(i in seq_along(indexes)){
    page_url <- sprintf(url, positions[p], indexes[i])
    
    test <- page_url %>%
      read_html %>%
      html_nodes("#playertable_0 td") %>%
      html_text() %>%
      str_trim(side = "both") %>%
      keep( ~ str_length(.x) > 0) %>% 
      matrix(byrow = T, ncol = 16)
    
    test_df <- as.data.frame(test[-1, ], stringsAsFactors = F)
    names(test_df) <- c("PLAYER, TEAM POS", "C/A", "PASS_YDS", "PASS_TD", "PASS_INT", "RUSH", "RUSH_YDS", "RUSH_TD", "REC", "REC_YDS", "REC_TD", "REC_TAR", "2PC", "FUML", "SPECIAL_TD", "PTS")
    test_df$position <- position_vec[p]
    fantasy_list[[p + i - 1]] <- test_df
  }
}

fantasy_df <- bind_rows(fantasy_list) %>%
  mutate_at(3:16, as.numeric) %>%
  group_by(position) %>%
  arrange(position, desc(PTS)) %>%
  mutate(position_rank = 1:n())

pos_ranks <- fantasy_df %>%
  mutate(ppw = PTS/16) %>%
  group_by(position) %>%
  slice(c(seq(12, 12 * 4, by = 12), seq(14, 14 * 4, by = 14)) + 1) %>%
  arrange(position, position_rank) %>%
  select(1, position, ppw, position_rank) %>%
  mutate(diff = ppw - lag(ppw))

top_half <- fantasy_df %>%
  mutate(ppw = PTS/16) %>%
  group_by(position) %>%
  slice(c(seq(6, 6 * 2, by = 6), seq(7, 7 * 2, by = 7)) + 1) %>%
  arrange(position, position_rank) %>%
  select(1, position, ppw, position_rank) %>%
  mutate(diff = ppw - lag(ppw))
