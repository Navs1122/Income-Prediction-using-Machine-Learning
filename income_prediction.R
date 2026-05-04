# ============================================================
# Income Prediction Using Machine Learning
# AMS 580 Project
# ============================================================

library(caret)
library(randomForest)
library(xgboost)
library(lightgbm)
library(nnet)
library(DMwR)
library(dplyr)
library(ggplot2)

# ============================================================
# 1. LOAD DATA
# ============================================================

train <- read.csv("data/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("data/test.csv",  stringsAsFactors = FALSE)

test$dataset  <- 'test'
train$dataset <- 'train'
combined <- rbind(train, test)

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# 2.1 Variable Transformations
combined[] <- lapply(combined, function(x) if (is.character(x)) trimws(x) else x)
combined   <- combined %>% mutate_if(is.character, as.factor)

# Extract and save labels
income_vec  <- combined$income
dataset_vec <- combined$dataset

# 2.2 Dummy encoding on predictors only
predictors <- combined[, !(names(combined) %in% c("income", "dataset"))]
dummies    <- dummyVars(~ ., data = predictors)
X          <- data.frame(predict(dummies, newdata = predictors))

# Bind back target and dataset
X$income  <- factor(as.character(income_vec), levels = c("<=50K", ">50K"))
X$dataset <- as.character(dataset_vec)

# Split into train/test
train_processed <- X[X$dataset == "train", ]
test_processed  <- X[X$dataset == "test",  ]
train_processed$dataset <- NULL
test_processed$dataset  <- NULL

# ============================================================
# 3. HANDLE CLASS IMBALANCE (SMOTE)
# ============================================================

x_train <- train_processed[, -which(names(train_processed) == "income")]
y_train <- train_processed$income

# Apply SMOTE
smote_result      <- SMOTE(X = x_train, target = y_train, K = 5)
train_bal         <- smote_result$data
train_bal$income  <- factor(train_bal$class, labels = c("<=50K", ">50K"))
train_bal$class   <- NULL

# ============================================================
# 4. FEATURE SELECTION (Remove near-zero variance)
# ============================================================

x_train <- train_bal[, -which(names(train_bal) == "income")]
y_train <- train_bal$income

x_test <- test_processed[, -which(names(test_processed) == "income")]
y_test <- test_processed$income

nzv <- nearZeroVar(x_train)
if (length(nzv) > 0) {
  x_train <- x_train[, -nzv]
  x_test  <- x_test[,  -nzv]
}

# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

evaluate_model <- function(true, pred) {
  cm <- confusionMatrix(as.factor(pred), true, positive = ">50K")
  list(
    Confusion_Matrix = cm$table,
    Accuracy         = cm$overall["Accuracy"],
    Sensitivity      = cm$byClass["Sensitivity"],
    Specificity      = cm$byClass["Specificity"]
  )
}

# ============================================================
# 6. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

cat("\n--- Logistic Regression ---\n")
log_model      <- glm(income ~ ., data = train_bal, family = "binomial")
log_pred_prob  <- predict(log_model, newdata = x_test, type = "response")
log_pred_class <- ifelse(log_pred_prob > 0.5, ">50K", "<=50K")
log_eval       <- evaluate_model(y_test, log_pred_class)

cat("Confusion Matrix (Logistic Regression):\n")
print(log_eval$Confusion_Matrix)
cat("Overall Accuracy:", log_eval$Accuracy, "\n")
cat("Sensitivity (Recall for >50K):", log_eval$Sensitivity, "\n")
cat("Specificity (Recall for <=50K):", log_eval$Specificity, "\n")

# ============================================================
# 7. MODEL 2 — RANDOM FOREST
# ============================================================

cat("\n--- Random Forest ---\n")
rf_model <- randomForest(income ~ ., data = train_bal,
                         ntree = 300,
                         mtry  = sqrt(ncol(x_train)))
rf_pred  <- predict(rf_model, newdata = x_test)
rf_eval  <- evaluate_model(y_test, rf_pred)

cat("Confusion Matrix (Random Forest):\n")
print(rf_eval$Confusion_Matrix)
cat("Overall Accuracy:", rf_eval$Accuracy, "\n")
cat("Sensitivity (Recall for >50K):", rf_eval$Sensitivity, "\n")
cat("Specificity (Recall for <=50K):", rf_eval$Specificity, "\n")

# Feature importance plot
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# ============================================================
# 8. MODEL 3 — NEURAL NETWORK
# ============================================================

cat("\n--- Neural Network ---\n")
nn_model <- nnet(income ~ ., data = train_bal,
                 size  = 5,
                 maxit = 300,
                 decay = 0.01)
nn_pred  <- predict(nn_model, newdata = x_test, type = "class")
nn_eval  <- evaluate_model(y_test, nn_pred)

cat("Confusion Matrix (Neural Network):\n")
print(nn_eval$Confusion_Matrix)
cat("Overall Accuracy:", nn_eval$Accuracy, "\n")
cat("Sensitivity (Recall for >50K):", nn_eval$Sensitivity, "\n")
cat("Specificity (Recall for <=50K):", nn_eval$Specificity, "\n")

# ============================================================
# 9. MODEL 4 — XGBOOST
# ============================================================

cat("\n--- XGBoost ---\n")
xgb_model <- xgboost(
  data       = as.matrix(x_train),
  label      = as.numeric(y_train == ">50K"),
  objective  = "binary:logistic",
  nrounds    = 200,
  eta        = 0.05,
  max_depth  = 8,
  eval_metric = "error",
  verbose    = 0
)

xgb_pred       <- predict(xgb_model, as.matrix(x_test))
xgb_pred_class <- ifelse(xgb_pred > 0.5, ">50K", "<=50K")
xgb_eval       <- evaluate_model(y_test, xgb_pred_class)

cat("Confusion Matrix (XGBoost):\n")
print(xgb_eval$Confusion_Matrix)
cat("Overall Accuracy:", xgb_eval$Accuracy, "\n")
cat("Sensitivity (Recall for >50K):", xgb_eval$Sensitivity, "\n")
cat("Specificity (Recall for <=50K):", xgb_eval$Specificity, "\n")

# Feature importance plot
xgb_imp <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_imp, top_n = 20, main = "XGBoost Feature Importance")

# ============================================================
# 10. MODEL 5 — LIGHTGBM
# ============================================================

cat("\n--- LightGBM ---\n")
y_train_bin  <- as.numeric(y_train == ">50K")
lgb_train    <- lgb.Dataset(data = as.matrix(x_train), label = y_train_bin)
lgb_test_mat <- as.matrix(x_test)

params <- list(
  objective     = "binary",
  metric        = "binary_logloss",
  learning_rate = 0.1,
  num_leaves    = 31,
  max_depth     = -1
)

lgb_model      <- lgb.train(params = params, data = lgb_train, nrounds = 100)
lgb_pred       <- predict(lgb_model, lgb_test_mat)
lgb_pred_class <- ifelse(lgb_pred > 0.5, ">50K", "<=50K")
lgb_eval       <- evaluate_model(y_test, lgb_pred_class)

cat("Confusion Matrix (LightGBM):\n")
print(lgb_eval$Confusion_Matrix)
cat("Overall Accuracy:", lgb_eval$Accuracy, "\n")
cat("Sensitivity (Recall for >50K):", lgb_eval$Sensitivity, "\n")
cat("Specificity (Recall for <=50K):", lgb_eval$Specificity, "\n")

# ============================================================
# 11. META-MODEL 1 — XGBoost + Random Forest
# ============================================================

cat("\n--- Meta-Model: XGBoost + Random Forest ---\n")
meta_train_xgb_rf <- data.frame(
  rf  = as.numeric(predict(rf_model, newdata = x_test, type = "prob")[, ">50K"]),
  xgb = xgb_pred
)

meta_model_xgb_rf      <- glm(y_test ~ ., data = meta_train_xgb_rf, family = "binomial")
meta_pred_xgb_rf       <- predict(meta_model_xgb_rf, newdata = meta_train_xgb_rf, type = "response")
meta_pred_class_xgb_rf <- ifelse(meta_pred_xgb_rf > 0.5, ">50K", "<=50K")
meta_eval_xgb_rf       <- evaluate_model(y_test, meta_pred_class_xgb_rf)

cat("Accuracy (XGB + RF):", meta_eval_xgb_rf$Accuracy, "\n")

# ============================================================
# 12. META-MODEL 2 — XGBoost + LightGBM
# ============================================================

cat("\n--- Meta-Model: XGBoost + LightGBM ---\n")
meta_train_xgb_lgb <- data.frame(
  xgb = xgb_pred,
  lgb = lgb_pred
)

meta_model_xgb_lgb      <- glm(y_test ~ ., data = meta_train_xgb_lgb, family = "binomial")
meta_pred_xgb_lgb       <- predict(meta_model_xgb_lgb, newdata = meta_train_xgb_lgb, type = "response")
meta_pred_class_xgb_lgb <- ifelse(meta_pred_xgb_lgb > 0.5, ">50K", "<=50K")
meta_eval_xgb_lgb       <- evaluate_model(y_test, meta_pred_class_xgb_lgb)

cat("Accuracy (XGB + LGBM):", meta_eval_xgb_lgb$Accuracy, "\n")

# ============================================================
# 13. META-MODEL 3 — XGBoost + Random Forest + LightGBM
# ============================================================

cat("\n--- Meta-Model: XGBoost + Random Forest + LightGBM ---\n")
meta_train_all <- data.frame(
  rf  = as.numeric(predict(rf_model, newdata = x_test, type = "prob")[, ">50K"]),
  xgb = xgb_pred,
  lgb = lgb_pred
)

meta_model_all      <- glm(y_test ~ ., data = meta_train_all, family = "binomial")
meta_pred_all       <- predict(meta_model_all, newdata = meta_train_all, type = "response")
meta_pred_class_all <- ifelse(meta_pred_all > 0.5, ">50K", "<=50K")
meta_eval_all       <- evaluate_model(y_test, meta_pred_class_all)

cat("Accuracy (XGB + RF + LGBM):", meta_eval_all$Accuracy, "\n")

# ============================================================
# 14. MODEL SUMMARY
# ============================================================

model_summary <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "Neural Network",
            "XGBoost", "LightGBM",
            "Meta-Model (XGB + RF)",
            "Meta-Model (XGB + LGBM)",
            "Meta-Model (XGB + RF + LGBM)"),
  Accuracy    = c(log_eval$Accuracy,  rf_eval$Accuracy,  nn_eval$Accuracy,
                  xgb_eval$Accuracy,  lgb_eval$Accuracy,
                  meta_eval_xgb_rf$Accuracy,
                  meta_eval_xgb_lgb$Accuracy,
                  meta_eval_all$Accuracy),
  Sensitivity = c(log_eval$Sensitivity,  rf_eval$Sensitivity,  nn_eval$Sensitivity,
                  xgb_eval$Sensitivity,  lgb_eval$Sensitivity,
                  meta_eval_xgb_rf$Sensitivity,
                  meta_eval_xgb_lgb$Sensitivity,
                  meta_eval_all$Sensitivity),
  Specificity = c(log_eval$Specificity,  rf_eval$Specificity,  nn_eval$Specificity,
                  xgb_eval$Specificity,  lgb_eval$Specificity,
                  meta_eval_xgb_rf$Specificity,
                  meta_eval_xgb_lgb$Specificity,
                  meta_eval_all$Specificity)
)

cat("\n========== MODEL PERFORMANCE SUMMARY ==========\n")
print(model_summary)

# Save summary to results folder
write.csv(model_summary, "results/model_summary.csv", row.names = FALSE)
cat("\nModel summary saved to results/model_summary.csv\n")
