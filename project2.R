## 1.Dataset and Package

# 1.1 Load packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

# 1.2 Load dataset

library(tidyverse)
library(dplyr)
url <- "https://github.com/yushinglui/graduate_admission/blob/master/datasets_admission.csv?raw=true"
admission <- read.csv(url)

## 2.Data exploration

# 2.1 General properties of the dataset.

head(admission)
summary(admission)

# 2.2 In the dataset, there are 500 rows and 9 columns.

dim(admission)

# 2.3 There are no NA in the dataset.

str(admission)
sum(is.na(admission))

# 2.4 The diagram shows the relation between GRE score and chance of admit.

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit))+geom_point()+geom_smooth()+ggtitle(
  "The correlation between GRE score and chances of admit")

# 2.5 The correlation between GRE score and chances of admit with TOEFL Score column.

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=TOEFL.Score))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with TOEFL Score column")

# 2.6 The correlation between GRE score and chances of admit with University rating column

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=University.Rating))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with University rating column")

# 2.7 The correlation between GRE score and chances of admit with SOP column

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=SOP))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with SOP column")

# 2.8 The correlation between GRE score and chances of admit with LOR column

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=LOR))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with LOR column")

# 2.9 The correlation between GRE score and chances of admit with CGPA column

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=CGPA))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with CGPA column")

# 2.10 The correlation between GRE score and chances of admit with research column

ggplot(admission,aes(x=GRE.Score,y=Chance.of.Admit,col=Research))+geom_point()+ggtitle(
  "The correlation between GRE score and chances of admit with research column")

# 2.11 Summarize for the correlation with all different conditions.

library(corrplot)
admission <- admission %>% select(
  GRE.Score,TOEFL.Score,University.Rating,SOP,LOR,CGPA,Research,Chance.of.Admit)
M <- cor(admission)
corrplot(M, method = "circle")

## 3. Machine learning algorithm

# 3.1 Data Partitioning

library(caret)
set.seed(1)
test_index <- createDataPartition(y = admission$Chance.of.Admit, times = 1, p = 0.5, list = FALSE)
train_set <- admission[-test_index,]
test_set <- admission[test_index,]

# 3.2 K-Nearest Neighbor

m_knn <- knn3(Chance.of.Admit~., data =train_set)
summary(m_knn)

pred <- predict(m_knn, newdata=test_set)
knn_rmse <- sqrt(mean((pred-train_set$Chance.of.Admit)^2))
rmse_results <- data_frame(method = "knn", RMSE = knn_rmse)
rmse_results

# 3.3 Decision Tree

library(rpart)
m_dt <- rpart(Chance.of.Admit~., data = train_set)
summary(m_dt)

pred<-predict(m_dt, newdata = test_set)
dt_rmse <- sqrt(mean((pred-test_set$Chance.of.Admit)^2))
rmse_results <- bind_rows(
  rmse_results, data_frame(method="Decision Tree", RMSE = dt_rmse))
rmse_results

# 3.4 Randomforest

library(randomForest)
m_rf <- randomForest(Chance.of.Admit~., data = train_set)

pred<-predict(m_rf,newdata = test_set)
rf_rmse <- sqrt(mean((pred-test_set$Chance.of.Admit)^2))
rmse_results <- bind_rows(
  rmse_results, data_frame(method="RandomForest", RMSE = rf_rmse))
rmse_results

# 3.5 Linear regression

m_lr <- lm(Chance.of.Admit~., data=train_set)
summary(m_lr)

pred <- predict(m_lr, newdata=test_set)
lr_RMSE <- sqrt(mean((pred-test_set$Chance.of.Admit)^2))
rmse_results <- bind_rows(
  rmse_results, data_frame(method = "Linear regression", RMSE = lr_RMSE))
rmse_results


