library(rvest)
library(dplyr)
library(stringr)
url <- "http://247sports.com/Season/%i-Football/Recruits?Page=%i"

years <- 2013:2017
pages <- 1:20

read_page <- function(page_url){
  page_html <- read_html(page_url)
  
  page_players <- page_html %>%
    html_nodes(".name a , .score") %>%
    html_text() %>%
    str_trim() %>%
    matrix(ncol = 3, byrow = T) %>%
    as.data.frame(stringsAsFactors = F)
  
  page_team <- page_html %>%
    html_nodes(".right .jsonly") %>%
    html_attr("alt")
  
  if(length(page_team) < 50){
    page_team <- c(page_team, rep("", 50 - length(page_team)))
  }
  
  page_players$School <- page_team
  return(page_players)
}

recruit_list <- list()
i <- 1
for(year in years){
  for(page in pages){
    page_url <- sprintf(url, year, page)
    page_df <- read_page(page_url)
    page_df$Year <- year
    page_df$Page <- page
    recruit_list[[i]] <- page_df
    i <- i + 1
  }
}

recruit_df <- bind_rows(recruit_list)
names(recruit_df)[1:3] <- c("Player", "None", "Rating")
recruit_df$None <- NULL
recruit_df <- recruit_df %>%
  mutate(School = str_replace_all(School, " Edit", ""),
         Rating = as.numeric(Rating))
library(ggplot2)
library(ggjoy)


recruit_df %>%
  group_by(School) %>%
  filter(n() > 5) %>%
  ungroup %>%
  mutate(School = reorder(School, Rating, median)) %>% 
  ggplot(aes(x = Rating, y = School)) +
  geom_joy(scale = 2, rel_min_height = .05) + 
  expappr::theme_expapp() + 
  scale_x_continuous(limits = c(.825, 1.01)) +
  xlab("Average Recruit Ranking") +
  ylab("") +
  ggtitle("CFB Recruiting Distributions", "Top 500 players from each 2013-2017 247 Rankings List")
