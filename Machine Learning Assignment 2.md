---
title: "Machine Learning Assignment 2"
output: html_document
---

###Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har)


###Include required libraries and set the environment

```r
library(caret)
library(randomForest)
library(doParallel)
```

###Load the data and clean up to keep only required variables
All the variables/reading associated with time, window or name of the participant are removed. All the variables with high NAs are removed(any variable having NA for this dataset has largely the NA values only)
The variables associated with "belt", "arm", "dumbbell" and "forearm" sensor readings are kept as predictors.


```r
activity <- read.csv("pml-training.csv")
submission <- read.csv("pml-testing.csv")
activity <- activity[,colSums(is.na(activity)) == 0]
activity <- activity[,colSums((activity=="")) == 0]
keepCols <- grep(paste(c("belt","arm","dumbbell","forearm"),collapse="|"),colnames(activity),value=T)
activity <- activity[,c(keepCols,"classe")]
```

###Dividing the data into training set and testing set

```r
set.seed(3452)
intrain <- createDataPartition(y=activity$classe,p=0.7,list=F)
training <- activity[intrain,]
testing <- activity[-intrain,]
```

###Training the model with 6 fold cross validataion
The model is trained with 6 fold repeated cross validation. To speed up the process parallel processing is used which can work well on multicore computers


```r
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)
tgrid = expand.grid(mtry=c(6))
tControl <- trainControl(method="repeatedcv",number =6,repeats=6,allowParallel=TRUE)
modFit <- train(classe~.,data=training,method="rf",trControl=tControl,tuneGrid=tgrid)
stopCluster(cluster)
print(modFit)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (6 fold, repeated 6 times) 
## 
## Summary of sample sizes: 11448, 11447, 11449, 11447, 11447, 11447, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD   
##   0.9936183  0.9919271  0.001754165  0.002219252
## 
## Tuning parameter 'mtry' was held constant at a value of 6
## 
```

###Run the model against for cross-validation against testing set.
Confusion matrix here provide us the out of sample error rate, the limited number of k folds might result in overfitting

```r
pred <- predict(modFit,newdata=testing)
cm <- confusionMatrix(pred,testing$classe)
print(cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    0 1131    7    0    0
##          C    1    2 1018   16    0
##          D    0    0    1  948    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9921, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9930   0.9922   0.9834   1.0000
## Specificity            0.9986   0.9985   0.9961   0.9998   1.0000
## Pos Pred Value         0.9964   0.9938   0.9817   0.9989   1.0000
## Neg Pred Value         0.9998   0.9983   0.9983   0.9968   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1922   0.1730   0.1611   0.1839
## Detection Prevalence   0.2853   0.1934   0.1762   0.1613   0.1839
## Balanced Accuracy      0.9990   0.9958   0.9941   0.9916   1.0000
```
###Predict the classe variable for 20 observation for submission(Results not printed)

```r
answers <- predict(modFit,submission)
```
