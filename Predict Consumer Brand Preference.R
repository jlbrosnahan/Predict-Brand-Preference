# This project seeks to predict which computer brand customers prefer: Acer (0) or Sony (1)
# Dataframe = complete (train/test), incomplete (test)
# y Value = brand

# checking structure
str(complete)

# checking descriptive stats
summary(complete)
summary(complete$brand)

# set seed
set.seed(123)

# preprocessing, no NAs
sum(is.na(complete))

# converting nominal variables elevel, car, and zipcode from numeric to factors
complete$brand <- as.factor(complete$brand)
complete$age <- as.integer(complete$age)
complete$elevel <- as.integer(complete$elevel)
complete$car <- as.integer(complete$car)
complete$zipcode <- as.integer(complete$zipcode)

str(complete)

# createDataPartition() 75% and 25%
index1 <- createDataPartition(complete$brand, p=0.75, list = FALSE)
train1 <- complete[ index,]
test1 <- complete[-index,]

# Check structure of trainSet
str(trainSet)

# setting cross validation
control <- trainControl(method = 'repeatedcv', 
                        number=10, 
                        repeats = 1)

# train and automatic tuning
rfFit3 <- train(brand~.,
                data = trainSet,
                method = 'rf',
                trControl=control,
                tuneLength = 1) 

rfFit3

# train and manual tuning
rfFit4 <- train(brand~.,
                data = trainSet,
                method = 'rf',
                trControl=control,
                tuneLength = 5) 

rfFit4


# rfFit3 mtry = 1 (BEST MODEL):
# Accuracy   Kappa    
# 0.9193152  0.8290663
# Tuning parameter 'mtry' was held constant at a value of 2


# rfFit4 mtry = 2:
#     Accuracy    Kappa
# 2   0.9172956  0.8246651
# 3   0.9174297  0.8247653
# 4   0.9172944  0.8243331
# 5   0.9158117  0.8211470
# 6   0.9136570  0.8164884
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 3

# plotting variable importance using ggplot
# salary and age most important, followed by credit
ggplot(varImp(rfFit3, scale=FALSE)) +
  ggtitle('Variable Importance of Top RF Model')


# predicting on testSet using optimal rfFit3 model
rfPreds <- predict(rfFit3, newdata = testSet)
View(rfPreds)

# predicting using type = 'prob' helps see prediction for each observation
rfProbs <- predict(rfFit3, newdata = testSet, type = 'prob')

# confusion matrix to see where it's right and where it's wrong
confusionMatrix(data = rfPreds, testSet$brand)

# postResample is the only way to test if it will do well in real world OR if overfitting
# rfFit3 is not overfitting, accuracy and kappas are both near .92 and .84
postResample(rfPreds, testSet$brand)

## awesome step! provides comparison of predictions to actual within same DF!
compare_rf <- data.frame(testSet,rfPreds) 

# summary gives count of predictions
summary(rfPreds)
plot(rfPreds)

df <- compare_rf %>%
  group_by(brand, rfPreds) %>%
  summarise(count=n())
df

compare_rf %>%
  count(brand)

compare_rf %>%
  count(rfPreds)

ggplot(compare_rf, aes(rfPreds)) +
  geom_bar()


############################################################################################
# making predictions using top model (rfFitr3) on new data = incomplete

# import dataset and checking structure
str(incomplete)

# set seed
set.seed(123)

# preprocessing, no NAs
sum(is.na(incomplete))

# converting variables to integers and factors
incomplete$brand <- as.factor(incomplete$brand)
incomplete$age <- as.integer(incomplete$age)
incomplete$elevel <- as.integer(incomplete$elevel)
incomplete$car <- as.integer(incomplete$car)
incomplete$zipcode <- as.integer(incomplete$zipcode)

# check structure of processed dataframe
str(incomplete)

# predicting on new data 'incomplete'
incompletePreds <- predict(rfFit3, newdata = incomplete)
str(incompletePreds)

# postResample on first 102 observations to determine how well model doing on test df
subset_incomplete <- incomplete %>% slice(1:103)
postResample(incompletePreds,subset_incomplete$brand)

## awesome step! this provides comparison of predictions to actual within the df
compare_incomplete <- data.frame(incomplete,incompletePreds) 

# exporting to excel
library(openxlsx)
write.xlsx(compare_incomplete,"IncompleteComparison.xlsx")

# summary of predictions
summary(incompletePreds)

plot(incompletePreds)

compare_incomplete %>%
  group_by(brand, incompletePreds) %>%
  summarise(count=n())

ggplot(compare_incomplete, aes(incompletePreds)) +
  geom_bar()

