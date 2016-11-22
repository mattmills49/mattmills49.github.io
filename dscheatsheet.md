---
layout: page
title: Data Science Cheat Sheet
subtitle: R Packages for the non-sexy parts of Data Science
---

### Data Validation / Testing

There are two recent packages that work on validating the data in a data 
frame and another package from Hadley that is more of a replacement for
`stopifnot`.

* [assertr](https://github.com/ropenscilabs/assertr) / [vignette](https://cran.r-project.org/web/packages/assertr/vignettes/assertr.html)

> The assertr package supplies a suite of functions designed to verify assumptions about data early in an analysis pipeline so that data errors are spotted early and can be addressed quickly.




```r
suppressPackageStartupMessages(library(dplyr))
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
```

```
## # A tibble: 3 × 2
##     cyl  avg.mpg
##   <dbl>    <dbl>
## 1     4 26.66364
## 2     6 19.74286
## 3     8 15.10000
```


* [validate](https://github.com/data-cleaning/validate) / [vignette](https://cran.r-project.org/web/packages/validate/vignettes/intro.html)


> The validate R-package makes it super-easy to check whether data lives up to expectations you have based on domain knowledge. It works by allowing you to define data validation rules independent of the code or data set.
 


```r
library(validate)
data(women)
cf <- check_that(women, height > 0, weight > 0, height/weight > 0.5)
cf
```

```
## Object of class 'validation'
## Call:
##     check_that(women, height > 0, weight > 0, height/weight > 0.5)
## 
## Confrontations: 3
## With fails    : 1
## Warnings      : 0
## Errors        : 0
```

```r
v <- validator(height > 0, weight > 0, height/weight > 0)
confront(women, v)
```

```
## Object of class 'validation'
## Call:
##     confront(x = women, dat = v)
## 
## Confrontations: 3
## With fails    : 0
## Warnings      : 0
## Errors        : 0
```

If you can include this in an automated data processing step it will help
with checking the underlying data behaves like you think it does. 

* [assertthat](https://github.com/hadley/assertthat)

> assertthat provides a drop in replacement for stopifnot() that makes it easy to check the pre- and post-conditions of a function, while producing useful error messages.



```r
library(assertthat)
x <- 1:10
stopifnot(is.character(x))
```

```
## Error: is.character(x) is not TRUE
```

```r
assert_that(is.character(x))
```

```
## Error: x is not a character vector
```

```r
assert_that(length(x) == 5)
```

```
## Error: length(x) not equal to 5
```

```r
assert_that(is.numeric(x))
```

```
## [1] TRUE
```

### Textbooks

* Data Analysis Using Regression and Multilevel/Hierarchical Models
* The Elements of Statistical Learning
* [Advanced Data Analysis from an Elementary Point of View](https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/ADAfaEPoV.pdf)
* [Regression Modeling Strategies](http://www.springer.com/gb/book/9781441929181)

