library("caret")
#data collection
setwd("E:/final_data")
data <- read.csv("final_data.csv", header=F)
#data preparation - adding column header 
header_name<-c("year", "month", "day", "atime", "placement_id", "exchange_id", "hour",
               "name", "exchange_name", "site_id", "site_name", "size", "target")
colnames(data) <- header_name
colnames(data)
#near zero variance checking
nzv <- nearZeroVar(data)
nzv
#just removed year
selected_header <- c("month", "day", "atime", "placement_id", "exchange_id",
                     "hour", "site_id", "size", "target")
data1 <- (data[,selected_header])
#View(data1)
dim(data1)
#to derive weekday 
x <- as.character((data$atime))
y <-strptime(x,"%Y-%m-%d %H:%M:%S", tz = "")
data1$weekday <- y$wday

data1$size <- as.character(data1$size)
selected_header_2 <- c("month", "day", "placement_id", "exchange_id",
                       "hour", "site_id", "size", "weekday", "target")
data2 <- (data1[,selected_header_2])

feature.names <- names(data2)[1:(ncol(data2)-1)]

#to change every char type to numeric equivalent
for (f in feature.names) {
  if (class(data2[[f]])=="character") {
    levels <- unique(data2[[f]])
    data2[[f]] <- as.integer(factor(data2[[f]], levels=levels))
    
  }
}
data2 <- data.frame(lapply(data2, as.numeric))
summary(data2)
#missing value - impute by -1 [ site id]
data2[is.na(data2)] <- -1

#CONVERT TARGET AS FACTOR WITH - YES - NO LABEL
data2$target <- as.factor(data2$target)
data2$target <- ifelse(data2$target==1,"YES", "NO")
data2$target <- as.factor(data2$target)
###data preparation completed


##data split - training and test
set.seed(1234)
train <- data2[sample(nrow(data2)),]
split <- floor(nrow(train)/2)
trainData <- train[0:split,]
testData <- train[(split+1):(split*2),]

str(trainData)

labelName <- 'target'
predictors <- names(train)[1:(ncol(train)-1)]

#checking the stats for target variable distribution for each set
table(trainData$target)
table(testData$target)

#sampling down - training set
EN_DATA_YES <- trainData[which(trainData$target=="YES"),]
EN_DATA_NO <- trainData[sample(nrow(trainData[which(trainData$target=="NO"),]),nrow(EN_DATA_YES)),]
balanced_train <- rbind(EN_DATA_YES, EN_DATA_NO)
balanced_train <- balanced_train[sample(nrow(balanced_train)),]


######
library("e1071")
m <- naiveBayes(target ~ ., data = balanced_train)
t1 <- table(predict(m, testData[,predictors]), testData[,9])
t1 <- as.data.frame.matrix(t1)
#t1[1,1]
accu_naive <-sum(t1[1,1] + t1[2,2])/sum(t1)
precision_naive <- t1[2,2]/sum(t1$YES)
accu_naive
precision_naive

library("randomForest")
rfm <- randomForest(target ~ ., data = balanced_train, ntry=3, ntree=25)
t <- table(predict(rfm, testData[,predictors]), testData[,9])
t <- as.data.frame.matrix(t)
#t[1,1]
accu_rf <-sum(t[1,1] + t[2,2])/sum(t)
precision_rf <- t[2,2]/sum(t$YES)
accu_rf 
precision_rf  
varImpPlot(rfm)


#####to provide probability of conversion of the data set####3
prob_score <- predict(rfm, data2[,predictors], "prob")
######################################################


library("caret")
myControl <- trainControl(method='cv', number=10, returnResamp='none')
#benchmark model - gbm
test_model <- train(balanced_train[,predictors], balanced_train[,labelName], method='gbm', trControl=myControl)
preds <- predict(object=test_model, testData[,predictors])

head(preds)
t <- table(preds, testData[,9])
t_gbm <- as.data.frame.matrix(t)
#t[1,1]
accu_gbm <-sum(t[1,1] + t[2,2])/sum(t)
precision_gbm <- t[2,2]/sum(t_gbm$YES)
accu_gbm
precision_gbm



set.seed(1234)
train <- data2[sample(nrow(data2)),]
split <- floor(nrow(train)/3)
ensembleData <- train[0:split,]
blenderData <- train[(split+1):(split*2),]
testingData <- train[(split*2+1):nrow(train),]


# train 3 the models with balanced_train data
model_gbm <- train(balanced_train[,predictors], balanced_train[,labelName], method='gbm', trControl=myControl)
model_rf <- train(balanced_train[,predictors], balanced_train[,labelName], method='rf', ntree=50)
model_rpart <- train(balanced_train[,predictors], balanced_train[,labelName], method='rpart', trControl=myControl)

# get predictions for each ensemble model for two last data sets
# and add them back to themselves
blenderData$gbm_PROB <- predict(object=model_gbm, blenderData[,predictors])
blenderData$rf_PROB <- predict(object=model_rf, blenderData[,predictors])
blenderData$rpart_PROB <- predict(object=model_rpart, blenderData[,predictors])


testingData$gbm_PROB <- predict(object=model_gbm, testingData[,predictors])
testingData$rf_PROB <- predict(object=model_rf, testingData[,predictors])
testingData$rpart_PROB <- predict(object=model_rpart, testingData[,predictors])

# see how each individual model performed on its own

## GBM performance
t <- table(testingData$gbm_PROB, testingData[,9])
t_gbm <- as.data.frame.matrix(t)
accu_gbm <-sum(t[1,1] + t[2,2])/sum(t)
precision_gbm <- t[2,2]/sum(t_gbm$YES)
accu_gbm
precision_gbm 


#RF -performance
t <- table(testingData$rf_PROB, testingData[,9])
t_rf <- as.data.frame.matrix(t)
accu_rf <-sum(t[1,1] + t[2,2])/sum(t)
precision_rf <- t[2,2]/sum(t_gbm$YES)
accu_rf 
precision_rf  


#Rpart -performance
t <- table(testingData$rpart_PROB, testingData[,9])
t_rpart <- as.data.frame.matrix(t)
accu_rpart <-sum(t[1,1] + t[2,2])/sum(t)
precision_rpart <- t[2,2]/sum(t_rpart$YES)
accu_rpart
precision_rpart 


predictors <- names(blenderData)[names(blenderData) != labelName]
BL_DATA_YES <- blenderData[which(blenderData$target=="YES"),]
head(BL_DATA_YES)
BL_DATA_NO <- blenderData[sample(nrow(blenderData[which(blenderData$target=="NO"),]),nrow(BL_DATA_YES)),]
balanced_blender <- rbind(BL_DATA_YES, BL_DATA_NO)
head(balanced_blender)


final_blender_model <- train(balanced_blender[,predictors], balanced_blender[,labelName], method='rf', ntree=25)

# See final prediction and performance of blended ensemble
preds <- predict(object=final_blender_model, testingData[,predictors])
t <- table(preds, testingData[,9])
t_rf <- as.data.frame.matrix(t)
accu_rf <-sum(t[1,1] + t[2,2])/sum(t)
precision_rf <- t[2,2]/sum(t_gbm$YES)
accu_rf 
precision_rf  
