# Detecting Diabetes in Patients based on Questionnaire Answers

**Abstract — According to a study conducted by the National Center for Health Statistics in 2020, diabetes was the eighth leading cause of death in the United States with 101,106 recorded deaths, a 15% increase from the prior year [2]. Diabetes is a chronic, life threatening disease that affects how the body processes sugar, and can sometimes be avoided by leading a healthy lifestyle [4]. Every year the Centers for Disease Control and Prevention caries out a phone survey reaching thousands of Americans inquiring them about their lifestyle, health risks, and their known diabetes diagnosis. The goal of this paper is to utilize multiple different machine learning models to predict if an individual has diabetes, or is prediabetic, based on their survey results. The research concludes that this dataset and the models implemented are not suitable for diagnosing adults with diabetes, and that a health professional and medical tests should be done to be properly diagnosed**

## Introduction
Diabetes is a chronic disease that affects how the body turns food into energy. The disease occurs when the pancreas does not produce enough insulin, a hormone that helps the body turn sugar into energy, or when the body cannot utilize insulin the body does produce [3]. Left untreated, excess sugar accumulates in the bloodstream which can result in more serious health diagnoses such as heart disease, vision loss, and kidney disease [4]. There are three types of diabetes, type 1, type 2, and gestational diabetes [4]; an individual can be diagnosed as prediabetic. Type 1 diabetes is an autoimmune reaction where the body destroys pancreas cells responsible for generating insulin, and only accounts for 5 - 10% of people who have diabetes [5]. Type 2 accounts for the majority of diabetes patients, about 90 - 95%, which occurs when the body can’t efficiently utilize insulin or maintain normal levels of blood sugar [4]. Type 2 diabetes can be prevented or delayed by adapting a healthy lifestyle, including losing weight, eating healthy, and staying active, meanwhile type 1 can’t be knowingly be prevented at this time [4]. Gestational diabetes develops in pregnant women without a history of diabetes, while this usually dissipates after women gives birth, it increases the chance of both the mother and child developing type 2 diabetes later in life [4]. Prediabetes is a health condition in which one’s blood sugar is higher than average, but not high enough to be diagnosed as type 2 diabetes [6]. Approximately every one in three adults in America are prediabetic, with about 80% being undiagnosed [4]. Further,
prediabetes increases the patients risk of developing type 2 diabetes, heart disease, and stroke [6]. Fortunately, if a patient is diagnosed prediabetic and takes the proper precautions, they can lower or even reverse their prediabetes diagnosis and severely reduce their risk of developing type 2 diabetes. This can be achieved if the patient adapts a healthy lifestyle which includes losing weight, and maintaining regular physical activity.

### Motivation and Problem Statement
In 2019, the Centers for Disease Control and Prevention (CDC) estimated there are roughly 37.3 million diagnosed, 8.5 million undiagnosed cases of diabetes, and 96 million cases of prediabetes among adults in the United States [3]. This translates to every one in ten adults having diabetes, and every one in three adults are prediabetic in America [7]. Diabetes poses a severe and abundant threat to the health of Americans. Proper diagnosis and awareness of existing diabetic and prediabetic cases is the first step to helping Americans reduce their risk of chronic illness. The goal of this project is to predict whether any given survey participant has diabetes, or is prediabetic, based on their survey answers. If the classifier is successful within certain parameters, the prediction results could help lower the number of undiagnosed diabetes cases in America and help medical professionals better understands what contributes to a diabetes diagnosis. Additionally, if an individual’s survey results are indicative that one might be high risk for having diabetes based on the model’s prediction and their survey answers, the participant’s model prediction results could provide suggestion for caller to see a medical professional for further testing. This system is not intended to replace professional medical advice or diagnosis, but may be able to serve as a preliminary method to aid participants in getting additional tests for diabetes.

### Dataset
The dataset of interest stems from the Behavioral Risk Factor Surveillance System (BRFSS) initiated by the CDC starting in 1984 to collect health data from Americans about their risk behaviours and health practices via phone calls [8]. The original source contains data from 1984 until 2021 with over 400,000 data points and 330 features for each year [8] The original dataset can be accessed through the CDC’s BRFSS website listed in the references section [8]. For the purpose of this study, only data from the year 2015 is utilized, and the dataset is sourced through Kaggle, an open-source dataset website [1]. The dataset can be accessed through Kaggle [here](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), or directly through [my repository](https://github.com/dgambone3/CSC6850_Machine_Learning_Project/tree/main/Diabeted%20Indicators%20Dataset). 
The dataset consists of 22 features that contain information about the callers health and lifestyle. The questionnaire inquires about certain health information such as whether their blood pressure and cholesterol levels are high, their body mass index (BMI), and if they have experienced heart disease or attack. The survey collects some identifying traits of the individual which include gender, age, education, and income. The dataset contains lifestyle information including average amount of fruits and vegetables consumed daily, whether or not they smoke, have had a stroke, been physically active in the past 30 days, and estimated alcohol consumption per week. The survey also collects the callers opinion of their general, mental and physical health as integers on different numeric scales. The data types for fifteen out of the twenty-two features are recorded as binary values, such that a 1 represents if the caller answered ’yes’, and 0 if the caller answered ’no’. The remaining features are integers representing a value on a scale or range of integers. The target feature has 3 variables, 0 for ’no diabetes’, 1 for ’prediabetes’, or 2 for ’diabetes’ if the caller does have diabetes.

## Methods
### Preprocessing
General data preprocessing and cleaning of the data was conducted in the preliminary steps in this research. The dataset was very clean initially, so there was little additional cleaning required. The dataset had zero null values, didn’t require any additional imputation, reduction, or transformation.
### Exploratory Data Analysis
When handling data that could be used in diagnosing the health of humans, all and any outliers could provide important information, specifically when looking at outliers of the negative, or positive diagnosis class. For this reason, it was decided to include all outliers as to not introduce additional bias into the dataset or models. Exploratory data analysis revealed that the dataset is unbalanced, which will affect how the following models will be implemented and is discussed in the following sections.
### Model Implementation
Multiple machine learning models are used for this research, along with three different ratios of training and test data in order to determine which model and data split best suits this research. The three different training and testing splits include 50/50, 70/30, 80/20 percents, respectively. A pipeline was constructed using a nested for-loop to fit and predict all models, conduct cross-validation, calculate metrics, and generate all outputs. This was done to streamline the process and ensure the data each model uses and all outputs are identical. The training and test data is split using SciKit-Learn’s train-test- split function from the model selection module. 10-fold cross validation is applied to the training data for each model, in which the best fold from each split is used to determine the best data split for each model based on lowest MSE. Learning curves are generated in a sepreate function for each model with all three data splits represented on one models’ plot in different color schemes. Confusion matrices are generated for the best split of all classification models. Classification models are measured by their accuracy, precision, recall, f1-score, MSE, learning curves, and confusion matrices. Regression models cannot be measured by the same metrics as classification models, and are measured by the MSE, R2 score, and learning curves in this study. Given the metrics for each models best split, the best overall model will be chosen by the user considering the models MSE, generalization displayed by each learning curve, and confusion matrices when applicable. All models are implemented using only model packaged provided in the SciKit-Learn library, for additional information on model specifics, the documentation for this library is listed in the references. [12]
The models included in this research are:
* Decision Tree Classifier
* Perceptron
* Multinomial Naive Bayes
* Logistic Regression
* Linear Regression
* SVM - Linear
* SVM - RBF
* Gradient Boost
* Muti-Layer Perceptron
* Regularilized Linear Regression
* Lasso Linear Regression
* k-Nearest Neighbors
* Linear Regression with Optimal Polynomial

## Results
Below are three tables summarizing the results of the best split for each model. The first table summarizes the MSE, split and calculated generalization for each model. The second summarizes all the metrics for the best split of each classification model. The final table summarizes the metrics for all regression models used in this research.

### Best Model Split
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/splits.png" width="500" />

### Classification Model Metrics
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/class_splits.png" width="500" />

### Regression Model Metrics
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/reg_splits.png" width="500" />




### Decision Tree
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/DT_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/DT_LC.png" width="500" />

### Perceptron
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/PERC_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/PERC_LC.png" width="500" />

### Multinomial Naive Bayes
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/MNB_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/MNB_LC.png" width="500" />

### Logistic Regression
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/LOGREG_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/LOGREG_LC.png" width="500" />

### Linear Regression
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/LINGRE_LC.png" width="500" />

### SVM - Linear
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/SVML_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/SVML_LC.png" width="500" />

### SVM - RBF
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/rbf_cm.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/SVM_RBF_LC.png" width="500" />

### Gradient Boost
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/GB_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/GB_LC.png" width="500" />

### Muti-Layer Perceptron
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/MLP_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/MLP_LC.png" width="500" />

### Regularilized Linear Regression
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/REGLINREG_LC.png" width="500" />

### Lasso Linear Regression
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/LASSO_LC.png" width="500" />

### k-Nearest Neighbors
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/KNN_CM.png" width="500" />
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/KNN_LC.png" width="500" />

### Linear Regression with Optimal Polynomial
<img src="https://github.com/dgambone3/CSC6850_Machine_Learning_Project/blob/main/images/POLY_LC.png" width="500" />

## Conclusion
Of all the classification models, it was decided that logistic regression and complement naive Bayes are the best options based on their ability to predict class 1 and 2 regardless if it resulted in lower metric scores. Based on the learning curve and metrics, the original linear regression model is the best. Out of all models, it is concluded that logistic regression is the best overall based on its prediction of class 1 and 2, and adequate metric scores, and learning curves.
Some intriguing discoveries during this research include that for many models the ’best’ was on a 50/50 split. It is generally known that it’s better to use more data for the training set than testing, essentially to give the model more instances of data to learn from. Thus it is concluded, that regardless of the metrics output, the user is responsible for making the best decision of certain metrics based on background knowledge and experience.
Some challenges were faced when it was discovered that initial plans for the research were not a possibility. For example in the early stages of this research, it was thought that a grid search would be able to be used for determining the best split and fold for each model. Due to the nature of the grid search function available, this ultimately wasn’t an option. SciKit-Learn’s GridSearchCV can only create searches on hyperparemeters for one model, not different instances of multiple models. Also, it was mistaken that accuracy and MSE would be good metrics for comparison between regression and classification models. This was resolved by realizing that MSE can be used for both classification and regression problems. It was unexpected that it would be difficult to compare the two model types. Though MSE was used, it felt as if the models were still difficult to compare. It was beneficial to be able to compare the confusion matrices for classification, but was difficult to continue comparing the models when the matrices were not an option for the regression models.
Though the ’best’ model is chosen, it is an important con- clusion that none of these models are suitable to diagnose
diabetes in adults based on their survey answers. All of the models would incorrectly predict participants who do have diabetes, or prediabetic, as someone without diabetes. Based on this data and model results, it would be unethical to use any of these models to diagnose disease in adults. It wouldn’t even be ethical to use any of these models as any sort of preliminary screening either. Diabetes is a serious disease, and it is recommended that diagnosis is only determined by a medical professional and laboratory tests, not a machine learning model based on questionnaire answers. This research proves how important it is to evaluate multiple metrics, mod- els, and variations of the dataset to get a good idea of how the models perform. For example, if a user were to look at only the weighted metrics for the classification models, they might think the model is fine. While in reality the model is very poor based on its learning curve and confusion matrix and should not be implemented for anything outside of research and education for machine learning problems.









### References 
[1] “Diabetes Health 08-Nov-2021. Indicators [Online]. Dataset,” Available: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators- dataset. [Accessed: 07-Mar-2023].

[2] F. B. Ahmad and R. N. Anderson, “The leading causes of death in the
US for 2020,” JAMA, vol. 325, no. 18, p. 1829, 2021.

[3] “National and State Diabetes Trends,” Centers for Disease Control and Prevention, 17-May-2022. [Online]. Available:
https://www.cdc.gov/diabetes/library/reports/reportcard/national-state-
diabetes-trends.html. [Accessed: 07-Mar-2023].

[4] “What is diabetes?,” Centers for Disease Control
and Prevention, 07-Jul-2022. [Online]. Available: https://www.cdc.gov/diabetes/basics/diabetes.html. [Accessed: 07- Mar-2023].

[5] “What is type 1 diabetes?,” Centers for Disease Con- trol and Prevention, 11-Mar-2022. [Online]. Available: https://www.cdc.gov/diabetes/basics/what-is-type-1-diabetes.html. [Accessed: 07-Mar-2023].

[6] “Prediabetes - your chance to prevent type 2 diabetes,” Centers for Disease Control and Prevention, 30-Dec-2022. [Online]. Avail- able: https://www.cdc.gov/diabetes/basics/prediabetes.html. [Accessed: 07-Mar-2023].

[7] “Prevalence of Both Diagnosed and Undiagnosed Diabetes,” Cen- ters for Disease Control and Prevention, 30-Sep-2022. [Online]. Available: https://www.cdc.gov/diabetes/data/statistics-report/diagnosed- undiagnosed-diabetes.html. [Accessed: 07-Mar-2023].

[8] “CDC - BRFSS - Survey Data and Documentation,” Centers for Disease Control and Prevention, 30-Sep-2021. [Online]. Avail- able: https://www.cdc.gov/brfss/data/documentation/index.html. [Ac- cessed: 08-Mar-2023].

[9] A. Dinh, S. Miertschin, A. Young, and S. D. Mohanty, “A data- driven approach to predicting diabetes and cardiovascular disease with Machine Learning - BMC Medical Informatics and Decision making,” BioMed Central, 06-Nov-2019. [Online]. Available: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911- 019-0918-5. [Accessed: 19-Apr-2023].

[10] “1.9. naive Bayes,” scikit. [Online]. Available: https://scikit- learn.org/stable/modules/naive bayes.html. [Accessed: 19-Apr-2023].

[11] J. Han, M. Kamber, and J. Pei, Data Mining: Concepts and Techniques, 3rd ed. Burlington, MA: Elsevier, 2012.

[12] “Scikit Learn API reference,” scikit. [Online]. Available: https://scikit- learn.org/stable/modules/classes.html. [Accessed: 22-Apr-2023].
