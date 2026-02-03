# FDA approval crosscheck (with partial/word-based matching) ----

# libraries ----
library(readxl)
library(dplyr)
library(stringr)
library(openxlsx)
library(tidyr)
library(purrr)

# read in v1 scores ----
v1_scores <- read_excel("./output/autoimmune_autoinflam_prediction_scores_normalized_080725.xlsx")
glimpse(v1_scores)

# FDA Approved Ever: if any TRUE for a drug name group ----
v1_scores <- v1_scores %>%
  group_by(`Drug Name`) %>%
  mutate(`FDA Approved Ever` = any(`FDA Approved` == TRUE, na.rm = TRUE)) %>%
  ungroup()

# counts ----
fda_approved_count <- sum(v1_scores$`FDA Approved` == TRUE, na.rm = TRUE)
fda_approved_ever_count <- sum(v1_scores$`FDA Approved Ever` == TRUE, na.rm = TRUE)

clean_drug <- function(x) {
  x %>%
    str_to_upper() %>%
    str_trim() %>%
    str_replace_all("[^A-Z0-9 ]", " ") %>%  # hyphens -> spaces
    str_squish()
}

base_token <- function(x_clean) {
  ifelse(is.na(x_clean) | x_clean == "", NA_character_, str_extract(x_clean, "^[A-Z0-9]+"))
}

# ---- Build FDA base-token lookup (from DrugName + ActiveIngredient) ----
fda_products <- read_excel("/Users/migonz/Desktop/Github_Projects/PrimeKG-TXGNN/data/Products_FDA.xlsx")

fda_base_set <- fda_products %>%
  transmute(
    DrugName_clean         = clean_drug(DrugName),
    ActiveIngredient_clean = clean_drug(ActiveIngredient)
  ) %>%
  pivot_longer(cols = c(DrugName_clean, ActiveIngredient_clean),
               values_to = "fda_clean") %>%
  filter(!is.na(fda_clean), fda_clean != "") %>%
  mutate(fda_base = base_token(fda_clean)) %>%
  filter(!is.na(fda_base), fda_base != "") %>%
  distinct(fda_base) %>%
  pull(fda_base)

# ---- Fast match on v1_scores ----
v1_scores <- v1_scores %>%
  mutate(
    drug_clean = clean_drug(`Drug Name`),
    drug_base  = base_token(drug_clean),
    FDA_match  = !is.na(drug_base) & (drug_base %in% fda_base_set)
  ) %>%
  select(-drug_clean, -drug_base)

# count number of TRUEs ----
fda_match_count <- sum(v1_scores$FDA_match == TRUE, na.rm = TRUE)

# arrange those 3 columns to be at end of the dataframe ----
v1_scores <- v1_scores %>%
  select(-`FDA Approved`, -`FDA Approved Ever`, -FDA_match,
         everything(),
         `FDA Approved`, `FDA Approved Ever`, FDA_match)

# rename columns ----
v1_scores <- v1_scores %>%
  rename(
    `FDA Approved for Specific Disease` = `FDA Approved`,
    `FDA Approved in Selected Autoimmune/Autoinflammatory Diseases` = `FDA Approved Ever`,
    `FDA Approved for Any Disease` = FDA_match
  )

# export ----
write.xlsx(
  v1_scores,
  "./output/autoimmune_autoinflam_prediction_scores_with_FDA_approval_020226.xlsx"
)

# (optional) print counts to console
cat("FDA Approved (specific disease) TRUE count:", fda_approved_count, "\n")
cat("FDA Approved Ever TRUE count:", fda_approved_ever_count, "\n")
cat("FDA Match (partial/word-based) TRUE count:", fda_match_count, "\n")
