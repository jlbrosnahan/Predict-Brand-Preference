## Predict-Brand-Preference
It's all about marketing, inventory needs, and maximizing profits through machine learning
* Machine learning binary classification
* Please see Predict-Brand.md for full analysis, scroll down for full business insights
* Also available on R Studio's RPubs at https://rpubs.com/brosnahj/659787

## Background
An Electronics Company would like for us to predict which computer brand customers from a new region will prefer, Acer or Sony. Doing so will help to determine inventory needs for new clientele and individual customer brand preferences for robust in-store and online marketing efforts. 

## Objective
The objective is to build predictive models and choose a model that can predict consumer computer brand preferene with at least 80% accuracy on test data. Ideal goal is a model that can predict brand preference with at least 90% level of certainty.

## Skills
* Modeling using manual and automatic tuning
* Analyzing performance metrics
* Business objectives met
* Actionable insights provided

## Top Model
* Random Forest 1 with an mtry of 2, 91.8% accuracy and 82.5% kappa, which is a useful determination of accuracy if predicted class is imbalanced, as it helps normalize an imbalance in the classes (in this problem the distribution is 62% to 38%).

## Actionable Insights
* Continue product inventory ratio of Sony (62%) to Acer (38%) for new customer clientele
* Target new customer base with direct-to-consumer email and mailing promotional efforts based on predicted brand preference
* Deploy algorithm in online platform for upselling and recommender strategies during all client online shopping experiences
