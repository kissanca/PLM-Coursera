# final project assignment Anca Maria Nagy
#Loading all the libraries and the data

library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(corrplot)
library(gbm)

set.seed(12345)

train_in <- read.csv("./data/pml-training.csv")
valid_in <- read.csv("./data/pml-testing.csv")

dim(train_in)

dim(valid_in)

#split the training set into a validation and sub training set.

trainData<- train_in[, colSums(is.na(train_in)) == 0]
validData <- valid_in[, colSums(is.na(valid_in)) == 0]
dim(trainData)

trainData<- train_in[, colSums(is.na(train_in)) == 0]
validData <- valid_in[, colSums(is.na(valid_in)) == 0]
dim(trainData)

trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]
dim(trainData)
dim(validData)

set.seed(12345) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]
dim(trainData)
dim(testData)

# Create and Test the Models 
NonZeroV <- nearZeroVar(trainData)
trainData <- trainData[, -NonZeroV]
testData  <- testData[, -NonZeroV]
dim(trainData)
dim(testData)

cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
highlyCorrelated = findCorrelation(cor_mat, cutoff=0.75)

names(trainData)[highlyCorrelated]

#Model building

# 1. Classification tree
set.seed(12345)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)

predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")

cmtree <- confusionMatrix(predictTreeMod1, as.factor(testData$classe))
cmtree

# plot matrix results
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))

# 2. Prediction with Random Forest

controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel

predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, as.factor(testData$classe))
cmrf
# plot matrix results
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel

predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, as.factor(testData$classe))
cmrf

plot(modRF1)
plot(cmrf$table, col = cmrf$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))

# 3. Generalized Boosted Regression Models

set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=trainData, method = "gbm", trControl = controlGBM, verbose = FALSE)
modGBM$finalModel

print(modGBM)

# validate and predict 

predictGBM <- predict(modGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, as.factor(testData$classe))
cmGBM

Output <- predict(modRF1, newdata=validData)
Output
