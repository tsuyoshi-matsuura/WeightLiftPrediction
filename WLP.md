---
title: "Distinguishing correct vs incorrect execution of weight lift exercises"
output:
  html_document:
    keep_md: yes
  pdf_document: default
editor_options:
  chunk_output_type: inline
---



## Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect
a large amount of data about personal activity relatively inexpensively. These type of
devices are part of the quantified self movement – a group of enthusiasts who take
measurements about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people regularly do is
quantify how *much* of a particular activity they do, but they rarely quantify how
*well* they do it.

The data for this project were collected from accelerometers on the belt, forearm, arm,
and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to use the collected
data to distinguish between the 5 types of execution of the exercises.

Two models were fit to the data, viz. a Random Forest and a Boosting model.
It was found that the Random Forest model was superior to the Boosting model
with an out of sample error of 1% vs 6% for the Boosting model.

## Data loading

The data used in this project are described on this website 
[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) 
(see the section on the Weight Lifting Exercise Dataset).

The training data were downloaded from the course website. While loading the data
the strings "", "NA" and "#DIV/0!" are marked as NA's


```r
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# Read training data
training <- read.csv(url(url_train),na.strings=c("","NA","#DIV/0!"))

# Size of the data set
dim(training)
```

```
## [1] 19622   160
```

Note that the training data has 19622 rows and 160 columns.

## Selection of predictors

### Data cleaning

Some exploratory data analysis showed that the columns with statistics of measurement, i.e.
columns with "kurtosis", "skewness", "max", "min", "amplitude", "var", "avg" or "stddev"
in their name, contained nearly only NA's. Only rows with "new_windows"
equal to "yes" had some non-NA values for the columns with statistics. As the number of
rows with "new_window=yes" was rather low (406 of the 19622 rows), it was decided
to remove all colums with statistics from the training data set.
Also the columns with the index 'X', 'user_name', timestamps and 'windows' information (columns 1-7)
were removed from the training data set, as they have no impact on the classification of the
weight lift exercise.


```r
# remove all statistics columns and index 'X', user_name, 'timestamp' and 'window' columns (1-7)
stat_cols <- grep("kurtosis|skewness|max|min|amplitude|var|avg|stddev",names(training))
stat_cols <- c(1:7,stat_cols)
training <- training[,-stat_cols]

dim(training)
```

```
## [1] 19622    53
```

By this data cleaning exercise the number of variables is reduced to 52, 13 for each of
the positions of the sensors (belt, arm, dumbbell and forearm).

### Predictor selection

A closer look at the 13 variables for each position of the sensors, shows that the they can
be divided into two groups, viz.:  

  1. variables labelled with 'roll', 'pitch', 'yaw' and 'total_accel'
  2. the remaining 9 variables which are lower level and directional
  
The variables in the first group are derived from the lower level ones in the second
group.

For building the prediction model only the variables in the first group are
used as predictors, because it is expected that the variables in the second
group are redundant. For this purpose the training set is further reduced.


```r
# Select columns with 'roll', 'pitch', 'yaw' and 'total_accel' in their name
col_drv <- grep("roll|pitch|yaw|total",names(training))
# Make sure to also include the classifier column 'classe', column 53
training <- training[,c(col_drv,53)]

dim(training)
```

```
## [1] 19622    17
```
With the above the number of predictors is reduced to 16.

## Data partitioning / validation set

The training set loaded earlier is split into a set for training the prediction model
and a validation set to test the model and to estimate the out of sample error.


```r
# Load caret library
library(caret)
# SPlit data into training and validation sets
set.seed(1234)
inTrain <- createDataPartition(y=training$classe,p=0.7,list=FALSE)
train <- training[inTrain,]
valid <- training[-inTrain,]
```

## Model building

Two models were built, one using Random Forest and another using Boosting. For
the tuning of the models k-fold cross-validation was implemented with k = 5.


```r
# set-up k-fold cross validation, k = 5
kfold <- trainControl( method ="cv", number=5)
```

### Random forest model

Below the Random Forest model is set-up with 5-fold cross-validation. An 
in-sample accuracy of around 99% is reported for the optimal model.


```r
# make Random Forest model
set.seed(5432)
modRF <- train(classe~.,data=train,method="rf", trControl=kfold)
print(modRF)
```

```
## Random Forest 
## 
## 13737 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10989, 10990, 10990, 10991 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9866056  0.9830558
##    9    0.9869701  0.9835188
##   16    0.9831849  0.9787320
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 9.
```

### Boosting model

Below the Boosting model is set-up with 5-fold cross-validation. An in-sample 
accuracy of around 93% is reported for the optimal model.


```r
# make Boosting model
set.seed(8725)
modGBM <- train(classe~.,data=train,method="gbm", trControl=kfold, verbose=FALSE)
print(modGBM)
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10989, 10989, 10991, 10990 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.6952016  0.6136781
##   1                  100      0.7503075  0.6843572
##   1                  150      0.7786251  0.7201552
##   2                   50      0.8037406  0.7518600
##   2                  100      0.8653985  0.8299347
##   2                  150      0.8926248  0.8642821
##   3                   50      0.8591374  0.8218858
##   3                  100      0.9124248  0.8892574
##   3                  150      0.9343371  0.9169753
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

## Model validation / Out of sample error

To estimate the out of sample error confusion matrices were calculated for
the two fitted models using the validation set that was defined earlier.
The results are shown below. The following out-of sample errors were found:

  * Random forest model: **1%**
  
  * Boosting model: **6%**
  
Conclusion is that the random forest model is more accurate than the boosting
model (at a 95% confidence level)

### Random forest model


```r
# Calculate confusion Matrix for validation set
confusionMatrix(predict(modRF,valid),valid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    7    0    0    0
##          B    3 1118   10    3    4
##          C    0   13 1011    7    3
##          D    0    1    5  953    0
##          E    0    0    0    1 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9903          
##                  95% CI : (0.9875, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9877          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9816   0.9854   0.9886   0.9935
## Specificity            0.9983   0.9958   0.9953   0.9988   0.9998
## Pos Pred Value         0.9958   0.9824   0.9778   0.9937   0.9991
## Neg Pred Value         0.9993   0.9956   0.9969   0.9978   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1900   0.1718   0.1619   0.1827
## Detection Prevalence   0.2851   0.1934   0.1757   0.1630   0.1828
## Balanced Accuracy      0.9983   0.9887   0.9903   0.9937   0.9967
```

### Boosting model


```r
# Calculate confusion Matrix for validation set
confusionMatrix(predict(modGBM,valid),valid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1634   42    3    6    0
##          B   17 1025   59   10   15
##          C   14   53  940   22   18
##          D    7   12   24  921   20
##          E    2    7    0    5 1029
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9429          
##                  95% CI : (0.9367, 0.9487)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9278          
##                                           
##  Mcnemar's Test P-Value : 2.363e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9761   0.8999   0.9162   0.9554   0.9510
## Specificity            0.9879   0.9787   0.9780   0.9872   0.9971
## Pos Pred Value         0.9697   0.9103   0.8978   0.9360   0.9866
## Neg Pred Value         0.9905   0.9760   0.9822   0.9912   0.9891
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2777   0.1742   0.1597   0.1565   0.1749
## Detection Prevalence   0.2863   0.1913   0.1779   0.1672   0.1772
## Balanced Accuracy      0.9820   0.9393   0.9471   0.9713   0.9741
```
