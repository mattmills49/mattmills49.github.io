library(rjson)
test <- fromJSON(file = "~/Documents/sample data/nba player tracking/0021500506.json")

event_lengths <- vapply(test$events, length, numeric(1))
# Each event has 4 lists; 
names(test$events[[1]])
# "eventId" "visitor" "home"    "moments"
#
names(test$events[[1]][[2]])
# "name"         "teamid"       "abbreviation" "players"   


library(doParallel)
# 8 cores available
cl <- makeCluster()
registerDoParallel(cl)

