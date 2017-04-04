#' # Using GAMs to Build a Strokes Gained Model for PGA Tour Golf
#' 
#' I recently participated in a Golf Hackathon put on by [15th
#' Club](http://www.15thclub.com/2017/03/02/15th-club-hackathon/), a company
#' involved in assisting Tour Professionals and Media Organizations use data
#' to enhance their play or products. It was a neat experience and opened my
#' eyes to some of the possibilities their data provides. With their permission
#' I will be presenting my initial attempt at building a Strokes Gained Model
#' based on the data the provided for the hackthon. Unfortunately I can not make
#' the data set public but if you are interested in access to the data perhaps
#' 15th Club themselves could provide it to you, who knows?
#' 
#' In this post I'll show how I used Generalized Additive Models to estimate
#' the shots remaining for a given shot and then the strokes gained once the 
#' shot has been completed. Generalized Additive Models allow us to assume each
#' variable's impact on our predictions is a smooth function which allows for
#' easy interpretation and visualization of the drivers of the model. 
#' 
#' ### The Shot Data
#' 
#+ shot_data, echo = F, warnings = F, messages = F
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(stringr))
options(dplyr.width = Inf)

file_path <- "~/Documents/sample data/hackathon-data-horsey.csv"

shots <- read_csv(file = file_path, col_types = cols(
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

#' 15th Club teamed up with Tour Player David Horsey to provide the hackathon 
#' competitors with 14 tournaments worth of shot data by Mr. Horsey. This 
#' included the type of club, the distance from the pin and the type of lie of
#' the ball both before and after each shot, as well as hole and tournament
#' information. In total we had 3412 shots for Mr. Horsey.
#' 
#' Obviously a proper Storkes Gained model would include shot data for multiple 
#' golfers across multiple years and tournaments with more detailed information.
#' But this is the data I was given and I think it's a rich enough data set to
#' try out some techniques as a first pass. 
#' 
#' ### Estimating the Strokes Remaining for a Hole
#' 
#' Before we find an estimate for how a shot improves the players performance on
#' a hole we first need a way to measure the current value of a golfer's lie
#' (lie here would include distance and the actual physical lie of the ball among
#' other data). Intuitavely a player on the fairway 100 yards from the green is
#' in much better position than someone in the rough 200 yards out, but we need
#' a way to quantify this. My initial idea was to build a model to estimate the 
#' number of strokes remaining on a hole given the player's current distance and
#' lie. Since the data contains the shot number of each shot and the score that 
#' Mr. Horsey got on each hole finding the number of shots he had left was
#' relatively trivial. 
#' 
#+ shot_remaining
shots <- shots %>%
  mutate(shots_remaining = score - shot_no + 1)

#' Before we start getting into actual modeling we need to settle on a distribution of the shots remaining on a hole for a
#' given lie before we can start modeling. Since the number of shots remaining
#' will never be negative we can't use a standard normal distribution so let's look at
#' how some distributions match up with the actual `shots_remaining` distribution. 
#'
#+ shots_remaining_distr, echo = F

set.seed(924516)
shots %>%
  select(shots_remaining) %>%
  mutate(poisson = rpois(n = n(), lambda = mean(shots_remaining)),
         normal = rnorm(n = n(), mean = mean(shots_remaining), sd = sd(shots_remaining)),
         lognormal = rlnorm(n = n(), meanlog = mean(log(shots_remaining)), sdlog = sd(log(shots_remaining)))) %>%
  gather(Distribution, Values) %>%
  mutate(Distribution = factor(Distribution, labels = c("Log-Normal", "Normal", "Poisson", "Actual Shots Remaining"))) %>%
  ggplot(aes(x = Values)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~ Distribution, nrow = 2) +
  scale_x_continuous(minor_breaks = seq(-1, 10), breaks = c(0, 4, 8)) +
  coord_cartesian(xlim = c(-1, 10)) +
  xlab("Shots Remaining") +
  ylab("Number of Observations") +
  ggtitle("Various Distributions for Estimating Shots Remaining") +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold", size = 10))

#' After seeing this distribution it struck me that the number of shots 
#' remaining for a hole wasn't a true counting distribution; Unless you hit a 
#' hole in one you can't have one shot left without at some point having two 
#' shots left and so on. In this sense the number of shots remaining is really 
#' seperate distributions of the golfer finishing the hole in one shot, two
#' shots, three shots, etc... We can model this easily using Ordinal Regression, 
#' a technique for modeling oredered categorical variables by finding a general
#' latent distribution and determining the cutoffs for different levels for this 
#' latent variable. 
#' 
#' So now that we know what model distribution we are going to use let's find out
#' what variables in our data could help us estimate the number of shots remaining
#' for a given shot's conditions. 
#' 
#' The distance from the pin seems like an obvious choice:
#+ distance_shots_remaining

ggplot(aes(x = left_to_pin_before / 36, y = shots_remaining), data = shots) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3, width = 0) +
  scale_color_discrete(guide = F) +
  xlab("Distance to Pin (Yards)") +
  ylab("Number of Shots Remaining") +
  ggtitle("Distance to the Pin's Impact on Shots Remaining") +
  theme_minimal()

#' The current lie is also probably a factor:
#+ lie_shots_remaining

shots %>%
  mutate(lie_before = str_replace_all(lie_before, " ", "\n")) %>%
  ggplot(aes(x = lie_before, y = shots_remaining)) +
  geom_jitter(aes(color = factor(shots_remaining)), height = .3) +
  scale_color_discrete(guide = F) +
  xlab("") +
  ylab("Number of Shots Remaining") +
  ggtitle("Current Lie's Impact on Shots Remaining") +
  theme_minimal() +
  theme(axis.text.x = element_text(face = "bold"))

#' The data does include some information about the hole itself like the distance
#' and par but I'll ignore those for now. 
#' 
#+ ord_gam

data_grid <- crossing(lie_before_factor = shots$lie_before_factor, left_to_pin_before = seq(1, 500) * 36)

lie_range <- shots %>%
  group_by(lie_before_factor) %>%
  summarize(p_lower = quantile(left_to_pin_before, probs = .05), p_upper = quantile(left_to_pin_before, probs = .95))

plot_df <- data_grid %>%
  left_join(lie_range, by = c("lie_before_factor")) %>% 
  filter(left_to_pin_before < p_upper, left_to_pin_before > p_lower)

shots$lie_before_factor <- as.factor(shots$lie_before) ## GAM needs by the by variable to be a factor
cats <- sort(unique(shots$shots_remaining))
shots$course_name <- factor(shots$course_id)

full_model <- gam(shots_remaining ~ s(course_name, bs = "re") + s(lie_before_factor, bs = "re") + s(left_to_pin_before, by = lie_before_factor),
                  data = shots,
                  family = ocat(theta = cats[c(-1, -8)]))

ordered_model <- gam(shots_remaining ~ s(left_to_pin_before, by = lie_before_factor),
                     data = shots,
                     family = ocat(theta = cats[c(-1, -8)]))

anova.gam(full_model, ordered_model)

mixed_test <- predict(full_model, type = "lpmatrix")

ordered_predictions <- predict.gam(full_model, newdata = mutate(data_grid, course_name = factor("new course")), type = "response")
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

 
