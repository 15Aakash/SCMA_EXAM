# Load necessary libraries
library(tidyverse)   # For data manipulation and visualization
library(caret)       # For machine learning utilities
library(rpart)       # For decision tree model
library(rpart.plot)  # For plotting decision trees
library(pROC)        # For ROC curves and AUC calculation
library(rsample)     # For data splitting
library(ggplot2)     # For enhanced plotting

# Load the dataset
bank_data <- read.csv("C:/Users/Aakash/Desktop/SCMA/bank-additional-full.csv", sep = ";")

# Check the first few rows of the dataset
head(bank_data)

# Check for missing values
total_missing_values <- sum(is.na(bank_data))
cat("Total missing values:", total_missing_values, "\n")

# Convert categorical variables to factors
bank_data <- bank_data %>%
  mutate(across(where(is.character), as.factor))

# Inspect the data structure
str(bank_data)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets using rsample
data_split <- initial_split(bank_data, prop = 0.7)
training_data <- training(data_split)
testing_data <- testing(data_split)

# Fit the logistic regression model
logistic_model <- glm(y ~ ., data = training_data, family = binomial)

# Predict on the test set using Logistic Regression
logistic_predictions_prob <- predict(logistic_model, testing_data, type = "response")
logistic_predictions_class <- ifelse(logistic_predictions_prob > 0.5, "yes", "no")

# Evaluate Logistic Regression Model
logistic_confusion_matrix <- confusionMatrix(as.factor(logistic_predictions_class), testing_data$y)
print(logistic_confusion_matrix)

# Calculate metrics
logistic_accuracy <- logistic_confusion_matrix$overall["Accuracy"]
logistic_precision <- logistic_confusion_matrix$byClass["Pos Pred Value"]
logistic_recall <- logistic_confusion_matrix$byClass["Sensitivity"]
logistic_f1_score <- 2 * ((logistic_precision * logistic_recall) / (logistic_precision + logistic_recall))

# AUC-ROC for Logistic Regression
logistic_roc_curve <- roc(testing_data$y, logistic_predictions_prob)
logistic_auc <- auc(logistic_roc_curve)

# Print metrics for Logistic Regression
cat("Logistic Regression Metrics:\n")
cat("Accuracy:", logistic_accuracy, "\n")
cat("Precision:", logistic_precision, "\n")
cat("Recall:", logistic_recall, "\n")
cat("F1 Score:", logistic_f1_score, "\n")
cat("AUC:", logistic_auc, "\n\n")

# Fit the decision tree model
decision_tree_model <- rpart(y ~ ., data = training_data, method = "class")

# Predict on the test set using Decision Tree
decision_tree_predictions_class <- predict(decision_tree_model, testing_data, type = "class")

# Evaluate Decision Tree Model
decision_tree_confusion_matrix <- confusionMatrix(decision_tree_predictions_class, testing_data$y)
print(decision_tree_confusion_matrix)

# Calculate metrics
decision_tree_accuracy <- decision_tree_confusion_matrix$overall["Accuracy"]
decision_tree_precision <- decision_tree_confusion_matrix$byClass["Pos Pred Value"]
decision_tree_recall <- decision_tree_confusion_matrix$byClass["Sensitivity"]
decision_tree_f1_score <- 2 * ((decision_tree_precision * decision_tree_recall) / (decision_tree_precision + decision_tree_recall))

# AUC-ROC for Decision Tree
decision_tree_predictions_prob <- predict(decision_tree_model, testing_data, type = "prob")[, 2]
decision_tree_roc_curve <- roc(testing_data$y, decision_tree_predictions_prob)
decision_tree_auc <- auc(decision_tree_roc_curve)

# Print metrics for Decision Tree
cat("Decision Tree Metrics:\n")
cat("Accuracy:", decision_tree_accuracy, "\n")
cat("Precision:", decision_tree_precision, "\n")
cat("Recall:", decision_tree_recall, "\n")
cat("F1 Score:", decision_tree_f1_score, "\n")
cat("AUC:", decision_tree_auc, "\n\n")

# Visualization

# Confusion Matrices
print(logistic_confusion_matrix)
print(decision_tree_confusion_matrix)

# Plot AUC-ROC Curves with ggplot2
# Plot AUC-ROC Curves
plot(roc_logistic, col = "blue", main = "AUC-ROC Curves", legacy.axes = TRUE)
plot(roc_tree, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lwd = 2)

# Plot the decision tree
rpart.plot(decision_tree_model, main = "Decision Tree Structure")

# Interpretation of Results

# Logistic Regression Coefficients
cat("Logistic Regression Coefficients:\n")
print(summary(logistic_model))
cat("Odds Ratios:\n")
print(exp(coef(logistic_model)))

# Decision Tree Structure
cat("Decision Tree Structure:\n")
print(decision_tree_model)
cat("Variable Importance:\n")
print(varImp(decision_tree_model))

