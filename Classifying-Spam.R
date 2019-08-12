# Load Packages ----
library(pacman)
p_load(
  "correlationfunnel",
  "gridExtra",
  "modelr",
  "readr",
  "SnowballC",
  "textfeatures",
  "tidymodels",
  "tidytext",
  "tidyverse"
)

# Import and Prepare the Data ----
spam <-
  read_csv("SPAM text message 20170820 - Data.csv")

# Cleaning the target variable
spam <- spam %>%
  mutate(Category = factor(ifelse(Category == "spam", 1, 0))) %>%
  rename("text" = Message)

# Feature engineering the text messages
n_vectors <- 0
spam_features <- spam %>%
  textfeatures(
    text,
    sentiment = FALSE,
    word_dims = FALSE,
    normalize = FALSE
  ) %>%
  bind_cols(spam)

# Explore the Data ----
# Checking for missing values
spam_features %>%
  map_df(~ sum(is.na(.))) %>%
  gather(key = "feature", value = "na_count") %>%
  arrange(desc(na_count))

# Plotting a correlation funnel
spam_features %>%
  select(-text) %>%
  binarize(thresh_infreq = 0.0001) %>%
  correlate(Category__X1) %>%
  plot_correlation_funnel()

# Create a simpler dataset with most correlated variables
spam_simple <- spam_features %>%
  select(
    Category,
    n_uq_chars,
    n_digits,
    n_caps,
    n_chars,
    n_lowersp,
    n_charsperword,
    n_uq_words,
    n_capsp,
    n_words,
    n_lowers
  )

# Ensure significance of correlated variables
spam_simple %>%
  select(-Category) %>%
  map(cor.test, as.numeric(spam_simple$Category))

# Visual the relationship between two most correlated variables
spam_simple %>%
  ggplot(aes(n_uq_chars, n_digits, color = Category)) +
  geom_point()

# Fit GLM Models to the Data ----
# Split the data for modeling
set.seed(123)
data_split <- spam_simple %>%
  initial_split(strata = "Category")

training_data <- training(data_split)
testing_data <- testing(data_split)

# Fit and measure the kappa of a model with two variables
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  metrics(truth, .pred_class)

# Observe the results of the model
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  conf_mat(truth, .pred_class)

logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data, type = "prob") %>%
  mutate(truth = as.numeric(testing_data$Category) - 1) %>%
  ggplot(aes(.pred_1, truth)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"))

plot1 <- testing_data %>%
  ggplot(aes(n_uq_chars, n_digits, color = Category)) +
  geom_point() +
  labs(title = "actual")

plot2 <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(n_uq_chars = testing_data$n_uq_chars,
         n_digits = testing_data$n_digits) %>%
  ggplot(aes(n_uq_chars, n_digits, color = .pred_class)) +
  geom_point() +
  labs(title = "glm")

grid.arrange(plot1, plot2, ncol = 1)

# Fit Random Forest Model to the Data ----
# Define grid of parameters
param_grid <- grid_regular(range_set(trees, c(50, 500)), levels = 25)

# Set random forest engine
rf_spec <- rand_forest("classification", trees = varying()) %>%
  set_engine("randomForest")

# Add model specifications to parameter grid
param_grid <- param_grid %>%
  mutate(specs = merge(., rf_spec))

# Define function for fitting models with grid of parameters
fit_one_spec <- function(model) {
  model %>%
    fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
    predict(new_data = testing_data) %>%
    mutate(truth = testing_data$Category) %>%
    kap(truth, .pred_class) %>%
    pull(.estimate)
}

# Fit model with each of the parameters and find optimal
param_grid %>%
  mutate(kappa = map_dbl(specs, fit_one_spec)) %>%
  filter(kappa == max(kappa))

# Observe results with the optimal parameter
rand_forest(trees = 68) %>%
  set_engine("randomForest") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  conf_mat(truth, .pred_class)

rand_forest(trees = 68) %>%
  set_engine("randomForest") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  metrics(truth, .pred_class)

plot3 <- rand_forest(trees = 68) %>%
  set_engine("randomForest") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(n_uq_chars = testing_data$n_uq_chars,
         n_digits = testing_data$n_digits) %>%
  ggplot(aes(n_uq_chars, n_digits, color = .pred_class)) +
  geom_point() +
  labs(title = "random forest")

grid.arrange(plot1, plot2, plot3, ncol = 2)