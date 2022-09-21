## ---------------------------
## Addressing Systematicity
## Author: Richard McAlexander
##
## Date Created: 2022-09-21
## Email: richardmcalexander@gmail.com
##
## This script contains functions, data manipulation, and code to produce plots for Research Data Scientist Technical Accessment

#clear workspace
rm(list=ls())

#load packages
library(tidyverse)
library(janitor)
library(tidymodels)

#import and clean data
df <- read_csv("data.csv") %>%
  clean_names()  %>%
  select(-ends_with("_id"),-application_attribute_1) %>%
  mutate(hired = ifelse(application_status=="hired",1,0),
         male = ifelse(gender=="Male",1,0),
         female = ifelse(gender == "Female",1,0),
         work_card = ifelse(candidate_demographic_variable_5=="work_card",1,0),
         citizenship = ifelse(candidate_demographic_variable_5=="citizenship",1,0),
         international_visa = ifelse(candidate_demographic_variable_5=="international_visa",1,0),
         work_permit = ifelse(candidate_demographic_variable_5=="work_permit",1,0),
         other_document = ifelse(candidate_demographic_variable_5=="other_document",1,0),
         white = ifelse(grepl("White",ethnicity),1,0),
         black = ifelse(grepl("Black",ethnicity),1,0),
         asian = ifelse(grepl("Asian",ethnicity),1,0),
         other = ifelse(grepl("other|Other",ethnicity),1,0),
         mixed = ifelse(grepl("Mixed|mixed",ethnicity),1,0)) %>%
  select(-application_status,-ethnicity,-gender,-candidate_demographic_variable_5,
         -candidate_interest_2) %>%
  replace(is.na(.), 0) %>%
  relocate(hired) %>%
  mutate(hired = as.factor(hired))

#we drop candidate interest_2 since they are all zeros

#split into training and test, and scale separately
test_index <- sample(1:nrow(df),10000,replace=FALSE)

df_train <- df[-test_index,] %>%
  mutate(number_of_employees_log = scale(number_of_employees_log),
         candidate_attribute_2 = scale(candidate_attribute_2),
         candidate_attribute_3 = scale(candidate_attribute_3),
         candidate_attribute_4 = scale(candidate_attribute_4),
         candidate_attribute_5 = scale(candidate_attribute_5),
         candidate_attribute_8 = scale(candidate_attribute_8),
         number_years_feature_1 = scale(number_years_feature_1),
         number_years_feature_2 = scale(number_years_feature_2),
         number_years_feature_3 = scale(number_years_feature_3),
         number_years_feature_4 = scale(number_years_feature_4),
         number_years_feature_5 = scale(number_years_feature_5),
         candidate_skill_1_count = scale(candidate_skill_1_count),
         candidate_skill_2_count = scale(candidate_skill_2_count),
         candidate_skill_3_count = scale(candidate_skill_3_count),
         candidate_skill_4_count = scale(candidate_skill_4_count),
         candidate_skill_5_count = scale(candidate_skill_5_count),
         candidate_skill_6_count = scale(candidate_skill_6_count),
         candidate_skill_7_count = scale(candidate_skill_7_count),
         candidate_skill_8_count = scale(candidate_skill_8_count),
         candidate_skill_9_count = scale(candidate_skill_9_count),
         candidate_skill_1_count = scale(candidate_skill_1_count),
         age = scale(age))

#normally we would scale the test set with reference to the training set, but for expediencies sake we will scale them separately.
df_test <- df[test_index,] %>%
  mutate(number_of_employees_log = scale(number_of_employees_log),
         candidate_attribute_2 = scale(candidate_attribute_2),
         candidate_attribute_3 = scale(candidate_attribute_3),
         candidate_attribute_4 = scale(candidate_attribute_4),
         candidate_attribute_5 = scale(candidate_attribute_5),
         candidate_attribute_8 = scale(candidate_attribute_8),
         number_years_feature_1 = scale(number_years_feature_1),
         number_years_feature_2 = scale(number_years_feature_2),
         number_years_feature_3 = scale(number_years_feature_3),
         number_years_feature_4 = scale(number_years_feature_4),
         number_years_feature_5 = scale(number_years_feature_5),
         candidate_skill_1_count = scale(candidate_skill_1_count),
         candidate_skill_2_count = scale(candidate_skill_2_count),
         candidate_skill_3_count = scale(candidate_skill_3_count),
         candidate_skill_4_count = scale(candidate_skill_4_count),
         candidate_skill_5_count = scale(candidate_skill_5_count),
         candidate_skill_6_count = scale(candidate_skill_6_count),
         candidate_skill_7_count = scale(candidate_skill_7_count),
         candidate_skill_8_count = scale(candidate_skill_8_count),
         candidate_skill_9_count = scale(candidate_skill_9_count),
         candidate_skill_1_count = scale(candidate_skill_1_count),
         age = scale(age))


# Begin Modeling ----------------------------------

#here we create a function to randomly re-order elements of the dataset as described in the report.
reorder_fun <- function(df,numcolumns=30,ratio=2){
  columns_to_flip <- sample(2:63,numcolumns,replace=FALSE)
  for (i in 1:numcolumns){
    rows_to_flip <- sample(1:nrow(df),(nrow(df)/ratio),replace=FALSE)
    df[rows_to_flip,columns_to_flip[i]] <- max(df[rows_to_flip
                                                  ,columns_to_flip[i]]) - df[rows_to_flip,columns_to_flip[i]]
  }
  return(df)
}

#this function outputs the predictions of a logit model trained on the re-ordered data 
get_preds <- function(){
  permuted <- reorder_fun(df_train)
  
  logit_output <- logistic_reg(
    mode = "classification",
    engine = "glm",
    penalty = 0,
    mixture = 0) %>%
    fit(hired ~ ., data = permuted)
  
  as.numeric(unlist(predict(logit_output,df_test))) -1
}


#This function replicates the function producing the prediction and outputs the share of model agreement (SMA) as a number.
share_consistent <- function(numreps){
  mypreds <- replicate(numreps,get_preds()) %>%
    as.data.frame()
  output <- (sum(rowSums(mypreds) == ncol(mypreds)|rowSums(mypreds)==0))/nrow(mypreds)
  return(output)

}

#create dataframe to store predictions to see how the number of iterations affects SMA
#looping or using sapply produces memory errors in Rstudio so we will just brute force this.
logit_pred_output_df <- data.frame(iterations=seq(5,50,by=5),
                                   predictions = NA)

logit_pred_output_df$predictions[1]<- share_consistent(logit_pred_output_df$iterations[1])
logit_pred_output_df$predictions[2]<- share_consistent(logit_pred_output_df$iterations[2])
logit_pred_output_df$predictions[3]<- share_consistent(logit_pred_output_df$iterations[3])
logit_pred_output_df$predictions[4]<- share_consistent(logit_pred_output_df$iterations[4])
logit_pred_output_df$predictions[5]<- share_consistent(logit_pred_output_df$iterations[5])
logit_pred_output_df$predictions[6]<- share_consistent(logit_pred_output_df$iterations[6])
logit_pred_output_df$predictions[7]<- share_consistent(logit_pred_output_df$iterations[7])
logit_pred_output_df$predictions[8]<- share_consistent(logit_pred_output_df$iterations[8])
logit_pred_output_df$predictions[9]<- share_consistent(logit_pred_output_df$iterations[9])
logit_pred_output_df$predictions[10]<- share_consistent(logit_pred_output_df$iterations[10])

#plot this output
plot_logit_agreement <- ggplot(logit_pred_output_df,aes(iterations,predictions))+
  geom_line()+
  geom_smooth()+
  xlab("Number of Logit Models")+
  ylab("Share of Agreement")+
  theme_light()

ggsave("plot_logit_agreement.png")
# Compare predictions across different types of models --------

#create datasets for each type of model, moving from all logits to RF, NN, and KNN
permutedf1_rf <- reorder_fun(df_train)
permutedf1_xg <- reorder_fun(df_train)
permutedf1_knn <- reorder_fun(df_train)

#train a random forest model
rf1_30 <- rand_forest(mode = "classification") %>%
  set_engine("ranger") %>%
  fit(hired ~ .,data = permutedf1_rf)

#train a xgboost
xg1_30 <- 
  boost_tree(mode = "classification",engine="xgboost") %>%
  fit(hired ~ ., data = permutedf1_xg)

#train a k-nearest neighbors
knn1_30 <- 
  nearest_neighbor(engine = "kknn") %>%
  set_mode("classification") %>% 
  fit(hired ~ ., data = permutedf1_knn)

#get and store predictions
df_pred_30 <- tibble(.rows = nrow(df_test)) 
df_pred_30$xg1_1_3 <- as.numeric(unlist(predict(xg1_30,df_test))) - 1
df_pred_30$rf1_1_3 <- as.numeric(unlist(predict(rf1_30,df_test))) -1 
df_pred_30$knn1_30 <- as.numeric(unlist(predict(knn1_30,df_test))) -1  

#find how many models agree
df_pred_30$total_one <- rowSums(df_pred_30)

#plot
plot_hist_agreement <- ggplot(df_pred_30)+
  geom_histogram(aes(total_one))+
  xlab("Number of Models With Same Prediction")+
  ylab("Number of Candidates")

ggsave("plot_hist_agreement.png")

#save
save.image("~/Dropbox/adept/renviron.RData")
