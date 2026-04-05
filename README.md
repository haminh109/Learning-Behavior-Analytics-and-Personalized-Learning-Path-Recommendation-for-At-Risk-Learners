# Learning Behavior Analytics and Personalized Learning Path Recommendation for At-Risk Learners

## Overview
This project analyzes student learning behavior on online learning platforms and develops a personalized learning path recommendation framework, with a focused study on learners who are at risk of dropping out.

Using the OULAD (Open University Learning Analytics Dataset), the project combines data preprocessing, exploratory data analysis, feature engineering, learner segmentation, at-risk prediction, recommendation design, and Power BI dashboarding to support data-driven educational decision-making.

## Project Objectives
The project is structured into two main parts:

### Part I. Learning Behavior Analytics and General Personalized Recommendation
- Analyze learning behavior patterns on an online learning platform
- Identify learner profiles based on engagement, progress, and performance
- Build a general personalized learning path recommendation mechanism
- Visualize insights and recommendations through Power BI dashboards

### Part II. Focused Study on At-Risk Learners
- Identify learners who are at risk of dropping out or failing to complete courses
- Analyze the behavioral factors associated with learning risk
- Develop a recommendation mechanism tailored to at-risk learners
- Propose early intervention insights for instructors and education managers

## Research Questions
1. What are the most important learning behavior patterns in online learning environments?
2. How do engagement, assessment performance, and learning activity affect learner outcomes?
3. Can learners be segmented into meaningful behavioral profiles?
4. How can personalized learning paths be recommended for different learner groups?
5. What early signals indicate that a learner is at risk?
6. Can machine learning models identify at-risk learners effectively?
7. How should recommendations for at-risk learners differ from general recommendations?

## Dataset
This project uses the **OULAD (Open University Learning Analytics Dataset)**.

Main tables used:
- `courses.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `vle.csv`
- `studentVle.csv`

## How to Clone This Repository with Full Data

This repository uses **Git LFS** for the large file `data/raw/studentVle.csv`.

To clone the repository with the full dataset, please install Git LFS first.

### Option 1: Fresh clone
git lfs install
git clone https://github.com/haminh109/Learning-Behavior-Analytics-and-Personalized-Learning-Path-Recommendation-for-At-Risk-Learners.git
cd Learning-Behavior-Analytics-and-Personalized-Learning-Path-Recommendation-for-At-Risk-Learners
git lfs pull

### Option 2: If you already cloned the repo
git lfs install
git lfs pull

**If Git LFS is not installed, large files such as studentVle.csv may appear only as pointer text files instead of the real dataset.**


## Project Workflow
1. Problem definition and research design
2. Data understanding and table mapping
3. Data cleaning and integration
4. Exploratory data analysis
5. Feature engineering
6. Learner segmentation
7. At-risk learner prediction
8. Recommendation design
9. Power BI dashboard development
10. Reporting and presentation

## Methods
### Data Preparation
- Missing value handling
- Duplicate checking
- Data type standardization
- Table joining and master dataset creation

### Exploratory Analysis
- Learner distribution by course and presentation
- Learning engagement over time
- Assessment performance analysis
- Comparison across learner outcome groups

### Feature Engineering
Examples of behavioral features:
- Total clicks
- Number of active days
- Recency of interaction
- Average assessment score
- Submission delay
- Early engagement level
- Learning persistence
- Learning risk index

### Learner Segmentation
- Rule-based segmentation
- K-Means clustering

### At-Risk Learner Identification
Candidate models:
- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Recommendation Design
The recommendation logic combines:
- Accessibility
- Behavioral similarity
- Engagement fit
- Completion probability

A general scoring idea:

Recommendation Score = w1 × Accessibility Score  
+ w2 × Similarity Score  
+ w3 × Engagement Fit Score  
+ w4 × Completion Probability

## Power BI Dashboard Design
The dashboard system is expected to include the following pages:

### Part I Dashboards
1. **Executive Overview**
2. **Learning Behavior Analytics**
3. **Personalized Recommendation for General Learners**

### Part II Dashboards
4. **At-Risk Learner Identification**
5. **At-Risk Behavior Deep Dive**
6. **Personalized Path Recommendation for At-Risk Learners**

## Expected Outputs
- Cleaned and integrated dataset
- Data dictionary
- EDA notebook
- Feature engineering notebook
- Segmentation and recommendation notebook
- At-risk prediction notebook
- Power BI dashboard file
- Final report
- Presentation slides

## Tech Stack
- **Python**: data preprocessing, EDA, feature engineering, machine learning
- **Power BI**: dashboarding and storytelling
- **GitHub**: version control and collaboration

Main Python libraries may include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Practical Value
This project is not only an academic exercise but also a practical decision-support framework for online learning platforms. It helps:
- understand learner behavior more clearly,
- detect at-risk learners earlier,
- recommend better learning paths,
- support instructors and managers with actionable insights.

## Authors
Final Project Team – Data Driven Marketing  
National Economics University

## License
This repository is for academic and educational purposes.
