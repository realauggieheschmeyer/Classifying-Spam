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

