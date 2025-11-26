# Load necessary libraries
# Ensure these are installed: install.packages(c("ggplot2", "car", "lmtest", "MASS", "corrplot", "ggcorrplot"))
library(ggplot2)
library(car)
library(lmtest)
library(MASS)
library(corrplot)

# ==========================================
# 1. Data Loading and Preparation
# ==========================================

# Read the dataset
data <- read.csv("prostate.csv")

# Inspect the structure
str(data)
summary(data)

# Convert 'train' to logical if it isn't already (CSV might read as string "TRUE"/"FALSE")
# In R, "TRUE"/"FALSE" strings usually need explicit conversion if not auto-detected
if(is.character(data$train)) {
  data$train <- as.logical(data$train)
}

# Split into Training and Testing sets based on the 'train' column
train_data <- subset(data, train == TRUE)
test_data  <- subset(data, train == FALSE)

# Remove the 'train' column as it is not a predictor
train_data <- subset(train_data, select = -train)
test_data  <- subset(test_data, select = -train)

# Check dimensions
cat("Training Set Dimensions:", dim(train_data), "\n")
cat("Testing Set Dimensions:", dim(test_data), "\n")

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================

# Correlation Matrix
# Compute correlation on numeric variables
cor_matrix <- cor(train_data)
print(round(cor_matrix, 2))

# Visualize Correlation
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         title = "Correlation Matrix (Training Data)", mar = c(0,0,1,0))

# Scatterplots of Predictors vs Target (lpsa)
# Reshape for plotting with ggplot2
library(reshape2)
train_long <- melt(train_data, id.vars = "lpsa")

ggplot(train_long, aes(x = value, y = lpsa)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", col = "blue", se = FALSE) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Scatterplots of lpsa vs Predictors", y = "Log PSA", x = "Predictor Value")

# ==========================================
# 3. Model Building: Full Model
# ==========================================

# Fit the full linear regression model on training data
full_model <- lm(lpsa ~ ., data = train_data)

# Summary of the full model
cat("\n--- Full Model Summary ---\n")
summary(full_model)

# Check for Multicollinearity
cat("\n--- Variance Inflation Factors (VIF) ---\n")
vif_values <- vif(full_model)
print(vif_values)
# Note: High VIFs (usually > 5 or 10) indicate problematic multicollinearity.

# ==========================================
# 4. Variable Selection
# ==========================================

# Stepwise selection based on AIC
# Direction "both" allows adding and dropping variables
step_model <- step(full_model, direction = "both", trace = 0)

cat("\n--- Selected Model Formula (Stepwise AIC) ---\n")
print(formula(step_model))

cat("\n--- Selected Model Summary ---\n")
summary(step_model)

# Compare AIC of full vs reduced model
cat("\nAIC Full Model:", AIC(full_model), "\n")
cat("AIC Selected Model:", AIC(step_model), "\n")

predictions <- predict(step_model, newdata = train_data)
actuals <- train_data$lpsa

comparison_df <- data.frame(Actual = actuals, Predicted = predictions)

ggplot(comparison_df, aes(x = Predicted, y = Actual)) +
  geom_point(color = "darkgreen", size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Log PSA (Train Set)",
       x = "Predicted lpsa", y = "Actual lpsa")

# ==========================================
# 5. Model Diagnostics (Selected Model)
# ==========================================

# 5.1 Residual Plots
par(mfrow = c(2, 2))
plot(step_model)
par(mfrow = c(1, 1))

# 5.2 Normality of Residuals
# Shapiro-Wilk Test
shapiro_test <- shapiro.test(residuals(step_model))
cat("\n--- Shapiro-Wilk Test for Normality ---\n")
print(shapiro_test)
# Null Hypothesis: Residuals are normally distributed. p < 0.05 indicates non-normality.

# 5.3 Heteroscedasticity
# Breusch-Pagan Test
bp_test <- bptest(step_model)
cat("\n--- Breusch-Pagan Test for Heteroscedasticity ---\n")
print(bp_test)
# Null Hypothesis: Homoscedasticity (constant variance). p < 0.05 indicates heteroscedasticity.

# 5.4 Influential Observations
# Cook's Distance
cooksD <- cooks.distance(step_model)
n <- nrow(train_data)
# Threshold often used is 4/n
influential_points <- which(cooksD > (4/n))
cat("\n--- Potential Influential Points (Cook's D > 4/n) ---\n")
print(influential_points)

# ==========================================
# Cook's Distance Plot Code
# ==========================================

# # 1. Calculate Cook's Distance for the stepwise model
# cooksD <- cooks.distance(step_model)
# 
# # 2. Define the threshold (commonly 4/n)
# n <- nrow(train_data)
threshold <- 4 / n

# 3. Create the plot
# 'pch = 20' makes small solid circles
# 'cex = 1' sets the size of the points
plot(cooksD, pch = 20, cex = 1, col = "darkblue",
     main = "Cook's Distance Plot for Influential Observations",
     ylab = "Cook's Distance", xlab = "Observation Index")

# 4. Add the threshold line
abline(h = threshold, col = "red", lty = 2, lwd = 2) # Dashed red line

# 5. Label the influential points
# Identify indices where Cook's D is greater than the threshold
influential_indices <- which(cooksD > threshold)

# Add text labels just above those points
# 'pos = 3' places text above the coordinate
text(x = influential_indices, 
     y = cooksD[influential_indices], 
     labels = influential_indices, 
     pos = 1, cex = 0.8, col = "red")

# # Optional: Add a legend
# legend("topright", legend = c("Cook's Distance", "Threshold (4/n)"),
#        col = c("darkblue", "red"), pch = c(20, NA), lty = c(NA, 2))

# ==========================================
# 6. Prediction and Validation
# ==========================================

# Predict on Test Data
predictions <- predict(step_model, newdata = test_data)
actuals <- test_data$lpsa

# Compute Performance Metrics
rmse <- sqrt(mean((predictions - actuals)^2))
mae <- mean(abs(predictions - actuals))
sst <- sum((actuals - mean(actuals))^2)
sse <- sum((predictions - actuals)^2)
r_squared_test <- 1 - (sse / sst)

cat("\n--- Test Set Performance ---\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE:", round(mae, 4), "\n")
cat("Test R-squared:", round(r_squared_test, 4), "\n")

# Plot Predicted vs Actual
comparison_df <- data.frame(Actual = actuals, Predicted = predictions)

ggplot(comparison_df, aes(x = Predicted, y = Actual)) +
  geom_point(color = "darkgreen", size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Log PSA (Test Set)",
       x = "Predicted lpsa", y = "Actual lpsa")

# ======================================================
# Advanced Modeling with R-Squared & RMSE Reporting
# ======================================================

# Load necessary libraries
library(MASS)   # For Robust Regression
library(glmnet) # For Ridge/Lasso
library(mgcv)   # For GAM

# Define actual values and SST (Total Sum of Squares) for the Test Set
# We need SST to calculate R-squared manually: R2 = 1 - (SSE / SST)
y_test_actual <- test_data$lpsa
sst_test <- sum((y_test_actual - mean(y_test_actual))^2)

# ------------------------------------------------------
# 1. Robust Regression (RLM)
# ------------------------------------------------------
robust_model <- rlm(lpsa ~ lcavol + lweight + age + lbph + svi + lcp + pgg45, 
                    data = train_data)

# Predict
pred_robust <- predict(robust_model, newdata = test_data)

# Calculate Metrics
sse_robust <- sum((y_test_actual - pred_robust)^2)
rmse_robust <- sqrt(mean((y_test_actual - pred_robust)^2))
r2_robust <- 1 - (sse_robust / sst_test)

cat("\n--- Robust Regression Performance ---\n")
cat("RMSE:", round(rmse_robust, 4), "| R-squared:", round(r2_robust, 4), "\n")


# ------------------------------------------------------
# 2. Lasso Regression (Regularization)
# ------------------------------------------------------
# Prepare Matrix Data
x_train_mat <- model.matrix(lpsa ~ ., train_data)[, -1]
y_train_vec <- train_data$lpsa
x_test_mat <- model.matrix(lpsa ~ ., test_data)[, -1]

# Fit Lasso
set.seed(123)
cv_lasso <- cv.glmnet(x_train_mat, y_train_vec, alpha = 1)
best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(x_train_mat, y_train_vec, alpha = 1, lambda = best_lambda)

# Predict
pred_lasso <- as.vector(predict(lasso_model, s = best_lambda, newx = x_test_mat))

# Calculate Metrics
sse_lasso <- sum((y_test_actual - pred_lasso)^2)
rmse_lasso <- sqrt(mean((y_test_actual - pred_lasso)^2))
r2_lasso <- 1 - (sse_lasso / sst_test)

cat("\n--- Lasso Regression Performance ---\n")
cat("RMSE:", round(rmse_lasso, 4), "| R-squared:", round(r2_lasso, 4), "\n")


# ------------------------------------------------------
# 3. Generalized Additive Model (GAM)
# ------------------------------------------------------
# Fit GAM (Smooth terms for continuous, linear for binary 'svi')
gam_model <- gam(lpsa ~ s(lcavol) + s(lweight) + s(age) + s(lbph) + svi + s(lcp) + s(pgg45), 
                 data = train_data)

# Predict
pred_gam <- predict(gam_model, newdata = test_data)

# Calculate Metrics
sse_gam <- sum((y_test_actual - pred_gam)^2)
rmse_gam <- sqrt(mean((y_test_actual - pred_gam)^2))
r2_gam <- 1 - (sse_gam / sst_test)

cat("\n--- GAM Performance ---\n")
cat("RMSE:", round(rmse_gam, 4), "| R-squared:", round(r2_gam, 4), "\n")


# ------------------------------------------------------
# Final Comparison Table
# ------------------------------------------------------
# Let's assume you have 'rmse' and 'r_squared_test' from the Stepwise OLS model earlier
# If not, ensure you run the Stepwise section from the previous code first.

# Create a data frame for clean output
results_table <- data.frame(
  Model = c("Stepwise OLS", "Robust Regression", "Lasso Regression", "GAM"),
  RMSE = c(rmse, rmse_robust, rmse_lasso, rmse_gam),
  R_Squared = c(r_squared_test, r2_robust, r2_lasso, r2_gam)
)

# Print the final comparison
cat("\n=========================================\n")
cat("      Final Model Performance Comparison   \n")
cat("=========================================\n")
print(results_table, digits = 4)

# ==========================================
# Combined Actual vs Predicted Plot (Side-by-Side 1x2)
# ==========================================
library(ggplot2)
library(tidyr)
library(dplyr)

# 1. Gather Predictions (Same as before)
# --------------------------------------
# For Lasso matrix
x_train_mat <- model.matrix(lpsa ~ ., train_data)[, -1]
x_test_mat  <- model.matrix(lpsa ~ ., test_data)[, -1]

train_results <- data.frame(
  Actual = train_data$lpsa,
  Stepwise = predict(step_model, newdata = train_data),
  Robust = predict(robust_model, newdata = train_data),
  Lasso = as.vector(predict(lasso_model, s = best_lambda, newx = x_train_mat)),
  GAM = predict(gam_model, newdata = train_data),
  Dataset = "Training Set"
)

test_results <- data.frame(
  Actual = test_data$lpsa,
  Stepwise = predict(step_model, newdata = test_data),
  Robust = predict(robust_model, newdata = test_data),
  Lasso = as.vector(predict(lasso_model, s = best_lambda, newx = x_test_mat)),
  GAM = predict(gam_model, newdata = test_data),
  Dataset = "Test Set"
)

# Combine and Pivot
all_results <- rbind(train_results, test_results) %>%
  pivot_longer(
    cols = c("Stepwise"),# "Robust", "Lasso", "GAM"),
    names_to = "Model",
    values_to = "Predicted"
  )

# 2. Generate the 1x2 Plot
# ------------------------
combined_plot_side_by_side <- ggplot(all_results, aes(x = Predicted, y = Actual, color = Model)) +
  # Points with transparency
  geom_point(alpha = 0.5, size = 1.5) +
  
  # Smooth regression lines for each model (no confidence bands for clarity)
  geom_smooth(method = "lm", se = FALSE, size = 0.8) +
  
  # Creates a 1 row x 2 columns layout
  facet_wrap(~Dataset, ncol = 2, scales = "free") +
  
  theme_minimal() +
  labs(
    title = "Actual vs Predicted Log PSA",
    # subtitle = "Comparison across models",
    x = "Predicted Log PSA",
    y = "Actual Log PSA",
    color = "Model Type"
  ) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

print(combined_plot_side_by_side)

# Save with WIDE dimensions
ggsave("combined_base.png",  #"combined_actual_vs_predicted_1x2.png", 
       plot = combined_plot_side_by_side, 
       width = 10, height = 5)
