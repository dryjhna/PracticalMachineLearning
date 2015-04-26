
###Include required libraries and set the environment

library(caret)
library(randomForest)
library(doParallel)

activity <- read.csv("pml-training.csv")
submission <- read.csv("pml-testing.csv")
activity <- activity[,colSums(is.na(activity)) == 0]
activity <- activity[,colSums((activity=="")) == 0]
keepCols <- grep(paste(c("belt","arm","dumbbell","forearm"),collapse="|"),colnames(activity),value=T)
activity <- activity[,c(keepCols,"classe")]

###Dividing the data into training set and testing set

set.seed(3452)
intrain <- createDataPartition(y=activity$classe,p=0.7,list=F)
training <- activity[intrain,]
testing <- activity[-intrain,]
```

The model is trained with 6 fold repeated cross validation. To speed up the process parallel processing is used which can work well on multicore computers

cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)
tgrid = expand.grid(mtry=c(6))
tControl <- trainControl(method="repeatedcv",number =6,repeats=6,allowParallel=TRUE)
modFit <- train(classe~.,data=training,method="rf",trControl=tControl,tuneGrid=tgrid)
stopCluster(cluster)
print(modFit)

pred <- predict(modFit,newdata=testing)
cm <- confusionMatrix(pred,testing$classe)
print(cm)

###Predict the classe variable for 20 observation for submission(Results not printed)

answers <- predict(modFit,submission)
