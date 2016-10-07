#' # Data Science Cheat Sheet
#'
#' ### Data Validation / Testing
#' 
#' There are two recent packages that work on validating the data in a data 
#' frame and another package from Hadley that is more of a replacement for
#' `stopifnot`.
#'
#' [**assertr**](https://github.com/ropenscilabs/assertr)
#' 
#' > The assertr package supplies a suite of functions designed to verify assumptions about data early in an analysis pipeline so that data errors are spotted early and can be addressed quickly.
#' 
#' Vignette: https://cran.r-project.org/web/packages/assertr/vignettes/assertr.html
#' 
#+ cache = T
library(assertr)
mtcars %>%
  verify(nrow(.) > 10) %>%
  verify(mpg > 0) %>%
  insist(within_n_sds(4), mpg) %>%
  assert(in_set(0,1), am, vs) %>%
  assert_rows(num_row_NAs, within_bounds(0,2), everything()) %>%
  insist_rows(maha_dist, within_n_mads(10), everything()) %>%
  group_by(cyl) %>%
  summarise(avg.mpg = mean(mpg))

#'
#' [**validate**](https://github.com/data-cleaning/validate)
#' 
#' > The validate R-package makes it super-easy to check whether data lives up to expectations you have based on domain knowledge. It works by allowing you to define data validation rules independent of the code or data set.
#' 
#' Vignette: https://cran.r-project.org/web/packages/validate/vignettes/intro.html
#' 
#+ cache = T
library(validate)
data(women)
cf <- check_that(women, height > 0, weight > 0, height/weight > 0.5)
summary(cf)

v <- validator(height > 0, weight > 0, height/weight > 0)
confront(women, v)

#' If you can include this in an automated data processing step it will help
#' with checking the underlying data behaves like you think it does. 
#' 
#' [**assertthat**](https://github.com/hadley/assertthat)
#' 
#' > assertthat provides a drop in replacement for stopifnot() that makes it easy to check the pre- and post-conditions of a function, while producing useful error messages.
#' 
#+ cache = T
library(assertthat)
x <- 1:10
stopifnot(is.character(x))

assert_that(is.character(x))

assert_that(length(x) == 5)

assert_that(is.numeric(x))

#ezknitr::ezspin(file = paste0(getwd(), "/raw_scripts/dscheatsheet.r"), out_dir = getwd(), keep_html = F, move_intermediate_file = T)
