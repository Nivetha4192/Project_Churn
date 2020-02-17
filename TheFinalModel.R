##################Required Libraries#######################
library(plyr)
library(corrplot)
library(ggplot2)
library(ROSE)
library(C50)
library(rpart)
library(caret)
library(corrgram)
library(randomForest)
library(gmodels)
library(gridExtra)
library(ROCR)


#given train and test datasets
df_train = read.csv('churn_train_data.csv')
df_test = read.csv('churn_test_data.csv')

#combining both given datasets 
customer_df = rbind(df_train,df_test)
str(customer_df)


#Removing the unwanted parameters
customer_df$phone.number <- NULL
customer_df$state <- NULL
customer_df$area.code <- NULL


########################EXPLORATORY DATA ANALYSIS##########################

#missing value analysis
sapply(customer_df,function(x)sum(is.na(x)))




###Boxplots to check for outliers in the data
ggplot(stack(customer_df), aes(x = ind, y = values)) +
  geom_boxplot() + coord_flip() 


#Variable Transformations
customer_df$Churn <- as.integer(customer_df$Churn)
customer_df$voice.mail.plan <- as.integer(customer_df$voice.mail.plan)
customer_df$international.plan <- as.integer(customer_df$international.plan)


customer_df$Churn[customer_df$Churn == '1'] <- 0
customer_df$Churn[customer_df$Churn == '2'] <- 1

customer_df$voice.mail.plan[customer_df$voice.mail.plan == '1'] <- 0
customer_df$voice.mail.plan[customer_df$voice.mail.plan == '2'] <- 1

customer_df$international.plan[customer_df$international.plan == '1'] <- 0
customer_df$international.plan[customer_df$international.plan == '2'] <- 1



#####Standard deviations
sapply(customer_df, sd)



#########################BI-VARIATE ANALYSIS#################
###########################Correlation Plot##################

corrgram(customer_df, order = F, lower.panel = panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
 
# Correlation with the Target Variable
ggplot(customer_df, aes(x=international.plan, y=Churn)) +
  geom_point(shape=1) +    
  geom_smooth(method=lm)

ggplot(customer_df, aes(x=total.day.minutes, y=Churn)) +
  geom_smooth(method=lm)

ggplot(customer_df, aes(x=total.day.charge, y=Churn)) +
  geom_smooth(method=lm)

ggplot(customer_df, aes(x=number.customer.service.calls, y=Churn)) +
  geom_smooth(method=lm)
#All these plots shows a positive relation with the Target Variable 




#Chi-Square Test for correlation with the Target Variable
chisq.test(customer_df$international.plan,customer_df$Churn)
chisq.test(customer_df$voice.mail.plan,customer_df$Churn)
#The p-Values are relatively less and says they are dependent on the Target Variable



#Target Class Distribution
barplot(prop.table(table(customer_df$Churn)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Target Class Distribution')
#There is a perfect target class imbalance problem




#############################MODEL DEVELOPMENT############################
#Train and test splits on the data
set.seed(1234)
indx <- sample(2,nrow(customer_df),replace = T, prob = c(0.7,0.3))
cdf_train <- customer_df[indx == 1,]
cdf_test <- customer_df[indx == 2,]

#Creating over,under,both and synthetic samples to overcome target imbalance
cdf_over = ovun.sample(Churn ~., data = cdf_train, method = 'over',N = 5984)$data
cdf_under = ovun.sample(Churn ~., data = cdf_train, method = 'under',N = 1004)$data
cdf_both = ovun.sample(Churn ~., data = cdf_train, method = 'both',
                       p = 0.5,
                       seed = 221,
                       N = 3494)$data

cdf_ROSE = ROSE(Churn ~., data = cdf_train,
                N = 5000,
                seed = 221)$data




                
#######################################
#      i) Decision Tree Model
#######################################
cdf_train$Churn <- as.factor(cdf_train$Churn)
cdf_test$Churn <- as.factor(cdf_test$Churn)
dt_model = C5.0(Churn ~ ., data = cdf_train,trials = 100, rules = FALSE)
summary(dt_model)

fit <- rpart(Churn ~ .,method="class", data=cdf_train)
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Churn")
text(fit, use.n=TRUE, all=TRUE, cex=.8  )





#Predictions with the Training Data
DT_pred = predict(dt_model, cdf_test, type = "class")

#ROC Curve
DT_roc = predict(dt_model,cdf_test,type = 'prob')[,2]
DT_roc =  prediction(DT_roc,cdf_test$Churn)
eval = performance(DT_roc,'acc')
plot(eval)


#Evaluating Model Performance using Confusion Matrix
cnf = table(cdf_test$Churn,DT_pred)
confusionMatrix(cnf) 
CrossTable(cdf_test$Churn,DT_pred,prop.c = F,prop.chisq = F,
           prop.r = F, dnn = c('actual default','predicted default') )

#########################################
#       ii) Random Forest Model
#########################################

RF_model = randomForest(Churn ~ ., cdf_train, importance = TRUE, ntree = 500)
importance(RF_model)

#Variable Importance 
plot.new()
varImpPlot(RF_model,type = 1)
abline(v = 45, col= 'blue')
#This plot resembles the important parameters in RF prediction

#Predict test data using random forest model
RF_Predictions = predict(RF_model, cdf_test)

#ROC Curve
RF_roc = predict(RF_model,cdf_test,type = 'prob')[,2]
RF_roc =  prediction(RF_roc,cdf_test$Churn)
eval_ = performance(RF_roc,'acc')
plot(eval_)



##Evaluate the performance of classification model
ConfMatrix_RF = table(cdf_test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

##########################################
#    iii) Logistic Regression Model 
##########################################

scaled_train = cdf_train
scaled_test = cdf_test
scaled_train[,1:17] = scale(scaled_train[,1:17])
scaled_test[,1:17] = scale(scaled_test[,1:17])


logit_model = glm(Churn ~ ., data = scaled_train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = scaled_test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, '1', '0')
logit_Predictions = as.factor(logit_Predictions)
logit_roc = predict(logit_model,cdf_test,type = 'prob')[,2]
logit_roc =  prediction(logit_roc,cdf_test$Churn)

##Evaluate the performance of classification model
ConfMatrix_LG = table(cdf_test$Churn, logit_Predictions)
confusionMatrix(ConfMatrix_LG)


#####################################
#       iv) KNN Implementation
#####################################
library(class)

#Predict test data
scaled_train[,1:17] = scale(scaled_train[,1:17])
scaled_test[,1:17] = scale(scaled_test[,1:17])
KNN_Predictions = knn(scaled_train[,-18], scaled_test[,-18], 
                      cl = scaled_train[,18], k = 5)


#Confusion matrix
Conf_matrix = table(scaled_test[,18], KNN_Predictions)
confusionMatrix(Conf_matrix)



############### END ################################
