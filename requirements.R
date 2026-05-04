# Install required R packages

packages <- c(
  "caret",
  "randomForest",
  "xgboost",
  "lightgbm",
  "nnet",
  "DMwR",      # for SMOTE
  "dplyr",
  "ggplot2"
)

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

invisible(lapply(packages, install_if_missing))
cat("All packages installed successfully.\n")
