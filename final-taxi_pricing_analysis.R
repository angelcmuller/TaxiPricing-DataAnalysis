library(readr)
library(rpart)
#install.packages("rpart.plot")
#install.packages("BART")
#install.packages("BayesTree")
#install.packages("randomForest")
#install.packages("gbm")
library(BayesTree)
library(rpart.plot)
library(BART)
library(gbm)

df <- read_csv("taxi_trip_pricing.csv")
summary(df)

# Confirming there are no NULL values in dataset
colSums(is.na(df))
nrow(df)

# Dealing with null values on target variable
df <- df[!is.na(df$Trip_Price), ]

### Handling null values on all other variables ###
df$Trip_Distance_km[is.na(df$Trip_Distance_km)] <- -1
df$Time_of_Day[is.na(df$Time_of_Day) | df$Time_of_Day == ""] <- "Unknown"
df$Day_of_Week[is.na(df$Day_of_Week) | df$Day_of_Week == ""] <- "Unknown"
df$Passenger_Count[is.na(df$Passenger_Count) | df$Passenger_Count == ""] <- -1
df$Traffic_Conditions[is.na(df$Traffic_Conditions) | df$Traffic_Conditions == ""] <- "Unknown"
df$Weather[is.na(df$Weather) | df$Weather == ""] <- "Unknown"
median_base_fare <- median(df$Base_Fare, na.rm = TRUE)
df$Base_Fare[is.na(df$Base_Fare)] <- median_base_fare
median_per_km_rate <- median(df$Per_Km_Rate, na.rm = TRUE)
df$Per_Km_Rate[is.na(df$Per_Km_Rate)] <- median_per_km_rate
median_per_minute_rate <- median(df$Per_Minute_Rate, na.rm = TRUE)
df$Per_Minute_Rate[is.na(df$Per_Minute_Rate)] <- median_per_minute_rate
median_trip_duration <- median(df$Trip_Duration_Minutes, na.rm = TRUE)
df$Trip_Duration_Minutes[is.na(df$Trip_Duration_Minutes)] <- median_trip_duration


### EDA ###
summary(df$Trip_Price)
hist(df$Trip_Price, breaks = 50, col = "skyblue", main = "Distribution of Trip Price", xlab = "Trip_Price")

hist(df$Trip_Distance_km, breaks = 50, col = "orange", main = "Trip Distance Distribution")
hist(df$Trip_Duration_Minutes, breaks = 50, col = "purple", main = "Trip Duration Distribution")

boxplot(Trip_Price ~ Day_of_Week, data = df, main = "Trip Price by Day of Week", col = "lightgreen")
boxplot(Trip_Price ~ Time_of_Day, data = df, main = "Trip Price by Time of Day", col = "lightblue")
boxplot(Trip_Price ~ Traffic_Conditions, data = df, main = "Trip Price by Traffic Condition", col = "brown")
boxplot(Trip_Price ~ Weather, data = df, main = "Trip Price by Weather", col = "yellow")

aggregate(Trip_Price ~ Weather, data = df, FUN = median)
aggregate(Trip_Price ~ Traffic_Conditions, data = df, FUN = median)
aggregate(Trip_Price ~ Time_of_Day, data = df, FUN = median)
aggregate(Trip_Price ~ Day_of_Week, data = df, FUN = median)


pairs(~ Trip_Price + Trip_Distance_km + Trip_Duration_Minutes + Base_Fare + Per_Km_Rate + Per_Minute_Rate + Passenger_Count, data = df)
 

### Building baseline regression tree ###
set.seed(123)

# Train-test split
n <- nrow(df)

# Random sample of 80% row indices
train_indices <- sample(seq_len(n), size = 0.8 * n)

# Split the data
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]


tree_model <- rpart(Trip_Price ~ ., data = train_data)

rpart.plot(tree_model, main = "Baseline Regression Tree")

## Making predictions ##
predictions <- predict(tree_model, newdata = test_data)

# Evaluate using RMSE and R-squared
rmse <- sqrt(mean((predictions - test_data$Trip_Price)^2))
r2 <- cor(predictions, test_data$Trip_Price)^2

cat("RMSE:", rmse, "\nR-squared:", r2, "\n")

summary(df$Trip_Price)
summary(df$Base_Fare)


## Bagging ##
library(randomForest)

num_predictors <- ncol(train_data) - 1
bag_model <- randomForest(Trip_Price ~ ., data = train_data,
                          mtry = num_predictors,
                          importance = TRUE)

bag_preds <- predict(bag_model, newdata = test_data)

y_test <- test_data$Trip_Price

# RMSE
bag_rmse <- sqrt(mean((bag_preds - y_test)^2))

# R-squared
bag_r2 <- cor(bag_preds, y_test)^2 

cat("Bagging RMSE:", bag_rmse, "\nBagging R-squared:", bag_r2, "\n")

varImpPlot(bag_model, main = "Variable Importance (Bagging)")



## Random Forest ##
set.seed(123)

rf_model <- randomForest(Trip_Price ~ ., data = train_data, importance = TRUE)

rf_preds <- predict(rf_model, newdata = test_data)

y_test <- test_data$Trip_Price

# RMSE
rf_rmse <- sqrt(mean((rf_preds - y_test)^2))

# R-squared
rf_r2 <- cor(rf_preds, y_test)^2

cat("Random Forest RMSE:", rf_rmse, "\nRandom Forest R-squared:", rf_r2, "\n")

varImpPlot(rf_model, main = "Variable Importance (Random Forest)")


## RF Model 2
set.seed(123)
rf_model_2 <- randomForest(Trip_Price ~ ., data = train_data,
                           mtry = 2,
                           importance = TRUE)
rf_preds_2 <- predict(rf_model_2, newdata = test_data)

rf_rmse_2 <- sqrt(mean((rf_preds_2 - y_test)^2))
rf_r2_2 <- cor(rf_preds_2, y_test)^2

cat("mtry = 2 ??? RMSE:", rf_rmse_2, "| R-squared:", rf_r2_2, "\n")

## RF Model 3
set.seed(123)
rf_model_6 <- randomForest(Trip_Price ~ ., data = train_data,
                           mtry = 6,
                           importance = TRUE)
rf_preds_6 <- predict(rf_model_6, newdata = test_data)

rf_rmse_6 <- sqrt(mean((rf_preds_6 - y_test)^2))
rf_r2_6 <- cor(rf_preds_6, y_test)^2

cat("mtry = 6 ??? RMSE:", rf_rmse_6, "| R-squared:", rf_r2_6, "\n")

## RF Model 4
set.seed(123)
rf_model_9 <- randomForest(Trip_Price ~ ., data = train_data,
                           mtry = 9,
                           importance = TRUE)
rf_preds_9 <- predict(rf_model_9, newdata = test_data)

rf_rmse_9 <- sqrt(mean((rf_preds_9 - y_test)^2))
rf_r2_9 <- cor(rf_preds_9, y_test)^2

cat("mtry = 9 ??? RMSE:", rf_rmse_9, "| R-squared:", rf_r2_9, "\n")


## Boosting ##

# Convert character columns to factors (in both train and test)
train_data[] <- lapply(train_data, function(col) {
  if (is.character(col)) as.factor(col) else col
})

test_data[] <- lapply(test_data, function(col) {
  if (is.character(col)) as.factor(col) else col
})


set.seed(123)

boost_model <- gbm(
  formula = Trip_Price ~ .,
  data = train_data,
  distribution = "gaussian",
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  verbose = FALSE
)

boost_model2 <- gbm(
  formula = Trip_Price ~ .,
  data = train_data,
  distribution = "gaussian",
  n.trees = 1000,
  interaction.depth = 5,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)

# Summary with variable importance of Model 1
summary(boost_model)
# Summary with variable importance of Model 2
summary(boost_model2)

boost_preds <- predict(boost_model, newdata = test_data, n.trees = 5000)
y_test <- test_data$Trip_Price

boost_rmse <- sqrt(mean((boost_preds - y_test)^2))
boost_r2 <- cor(boost_preds, y_test)^2

cat("Boosting RMSE:", boost_rmse, "\nBoosting R-squared:", boost_r2, "\n")

boost_preds2 <- predict(boost_model2, newdata = test_data, n.trees = 1000)
y_test2 <- test_data$Trip_Price

boost_rmse2 <- sqrt(mean((boost_preds2 - y_test2)^2))
boost_r22 <- cor(boost_preds2, y_test2)^2

cat("Boosting RMSE:", boost_rmse2, "\nBoosting R-squared:", boost_r22, "\n")



### BART ###
# 1. Prepare the data
# Drop the target variable 'Trip_Price' from predictors
x_train <- train_data[, !(names(train_data) %in% "Trip_Price")]
y_train <- train_data$Trip_Price

x_test <- test_data[, !(names(test_data) %in% "Trip_Price")]
y_test <- test_data$Trip_Price

# Convert character columns to factors in both train and test
x_train[] <- lapply(x_train, function(col) {
  if (is.character(col)) as.factor(col) else col
})

x_test[] <- lapply(x_test, function(col) {
  if (is.character(col)) as.factor(col) else col
})

# Convert training and test data to model matrices (removing intercept column)
x_train_matrix <- model.matrix(~ ., data = x_train)[, -1]
x_test_matrix <- model.matrix(~ ., data = x_test)[, -1]


library(BayesTree)

# 2. Fit the BART model
set.seed(123)
bart_model <- gbart(x.train = x_train_matrix, y.train = y_train, x.test = x_test_matrix)

# 3. Extract predictions
bart_preds <- bart_model$yhat.test.mean


# RMSE
bart_rmse <- sqrt(mean((bart_preds - y_test)^2))

# R-squared
bart_r2 <- cor(bart_preds, y_test)^2

cat("BART RMSE:", bart_rmse, "\nBART R-squared:", bart_r2, "\n")

