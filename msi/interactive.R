# Libraries
library(tidyverse)
library(caret)
library(parallel)
library(tictoc)
library(doParallel)
library(haven)
library(RANN)

# Data Import and Cleaning
gss <- read_sav("GSS2006.sav") %>% # read from the same folder
  select(BIG5A1,BIG5B1,BIG5C1,BIG5D1,BIG5E1,BIG5A2,BIG5B2,BIG5C2,BIG5D2,BIG5E2,HEALTH) %>%
  mutate_all(as.numeric) 
gss_tbl <- gss %>%
  filter(rowSums(is.na(gss[,1:10]))!=ncol(gss[,1:10])) %>%
  filter(!is.na(HEALTH))

# Data Analysis--ML xgbLinear Model
## Non-parallel
tic()
xgb_model <- train(
  HEALTH~.^3,
  data=gss_tbl,
  method="xgbLinear",
  tuneLength=3,
  trControl=trainControl(method="cv",number=10,verboseIter=T),
  preProcess=c("center","scale","zv","knnImpute"),
  na.action=na.pass
)
exec_time_np <- toc()

## Parallel
local_cluster <- makeCluster(2)
registerDoParallel(local_cluster)
tic()
xgb_model_p <- train(
  HEALTH~.^3,
  data=gss_tbl,
  method="xgbLinear",
  tuneLength=3,
  trControl=trainControl(method="cv",number=10,verboseIter=T),
  preProcess=c("center","scale","zv","knnImpute"),
  na.action=na.pass
)
exec_time_p <- toc()
stopCluster(local_cluster)

# The first run used 1 processor and took 203.3 seconds. Parallel processing used 3 preprocessors and took 132.3 seconds (71 seconds faster).
# In MSI, the first run took 471 seconds and parallel processing took 239 seconds. 

# Save Execution times
interactive <- data.frame("run"=c("non-parallel","parallel"),
                          "time"=c(exec_time_np$toc-exec_time_np$tic,exec_time_p$toc-exec_time_p$tic))
write_csv(interactive,"interactive.csv")
