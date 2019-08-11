# Load Packages ----
library(pacman)
p_load(
  "correlationfunnel",
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

# Explore the relationship between two most correlated variables
spam_simple %>%
  ggplot(aes(n_uq_chars, n_digits, color = Category)) +
  geom_point()

# Split the data for modeling
set.seed(123)
data_split <- spam_simple %>%
  initial_split(strata = "Category")

training_data <- training(data_split)
testing_data <- testing(data_split)

# Fit Models to the Data ----
# Fit and measure the kappa of a single-variable model
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  kap(truth, .pred_class)
# kappa = .736

# Fit and measure the kappa of a model with two variables
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  kap(truth, .pred_class)
# kappa = .857

# Fit and measure the kappa of a model with three variables
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ n_uq_chars + n_digits + n_caps, data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  kap(truth, .pred_class)
# kappa = .857

# Fit and measure the kappa of a model with all variables
logistic_reg() %>%
  set_engine("glm") %>%
  fit(Category ~ ., data = training_data) %>%
  predict(new_data = testing_data) %>%
  mutate(truth = testing_data$Category) %>%
  kap(truth, .pred_class)
# kappa = .899

# map(spam_simple[,2:11], shapiro.test)
# shapiro.test(spam_simple$n_uq_chars)