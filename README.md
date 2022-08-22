# Churn-Telco Classification Project

## Project Summary
<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create modules (acquire.py, prepare.py) that make your process repeateable and your report (notebook) easier to read and follow.
> - Ask exploratory questions of your data that will help you understand more about the attributes and drivers of customers churning. Answer questions through charts and statistical tests.
> - Construct a model to predict customer churn using classification techniques.
> - Refine your work into a Report, in the form of a jupyter notebook, that you will walk through in a 5 minute presentation to a group of collegues and managers about the work you did, why, goals, what you found, your methdologies, and your conclusions.
> - ABe prepared to answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals
> - Find drivers for customer churn at Telco. Why are customers churning?
> - Construct a ML classification model that accurately predicts customer churn.
> - Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?

#### Audience
> - Your target audience for your notebook walkthrough is your direct manager and their manager. This should guide your language and level of explanations in your walkthrough.

#### Project Deliverables
> - This Readme.md
> - A final report notebook  (.ipynb)
> - Acquire & Prepare Modules (.py)
> - Predictions (.csv)
> - A final report notebook presentation

#### Project Context
> - The Telco dataset I'm using came from the Codeup database.

#### Data Dictionary

|Target                    |Datatype               |Definition|
|:-------                  |:--------              |:----------|
| churn                    | 7043 non-null: object | churn - yes, no |
|                          |                       | 
| **Feature**              | **Datatype**          | **Definition** |
| payment_type_id          |7043 non-null: int64   | payment type encoded |
| internet_service_type_id |7043 non-null: int64   | internet service type encoded |
| contract_type_id         |7043 non-null: int64   | contract type encoded |
| customer_id              |7043 non-null: object  | unique key for customers |
| gender                   |7043 non-null: object  | gender - M/F |
| senior_citizen           |7043 non-null: int64   | senior citizen - hot encoded |
| partner                  |7043 non-null: object  | partner - Y/N |
| dependents               |7043 non-null: object  | dependents - Y/N |
| tenure                   |7043 non-null: int64   | tenure - count by months |
| phone_service            |7043 non-null: object  | phone_service - Y/N |
| multiple_lines           |7043 non-null: object  | multiple_lines - Y/N |
| online_security          |7043 non-null: object  | online_security - Y/N |
| online_backup            |7043 non-null: object  | online_backup - Y/N |
| device_protection        |7043 non-null: object  | device_protection - Y/N |
| tech_support             |7043 non-null: object  | tech_support - Y/N |
| streaming_tv             |7043 non-null: object  | streaming_tv - Y/N |
| streaming_movies         |7043 non-null: object  | streaming_movies - Y/N |
| paperless_billing        |7043 non-null: object  | paperless_billing - Y/N |
| monthly_charges          |7043 non-null: float64 | monthly charge amount |
| total_charges            |7043 non-null: object  | total charged to customer |
| churn                    |7043 non-null: object  | churn - Y/N |
| contract_type            |7043 non-null: object  | categorical, string for encoded |
| internet_service_type    |7043 non-null: object  | categorical, string for encoded |
| payment_type             |7043 non-null: object  | categorical, string for encoded |

#### Initial Hypotheses

> - **Hypothesis 1 -**
> - alpha = .05
> - $H_0$: Is churn independant of payment type. $\mu_{churn}$ independent $\mu_{payment type}$ (pval > alpha).  
> - $H_a$: Rejection of Null $\mu_{virginica}$ ~~independent~~  $\mu_{versicolor}$ (pval <= alpha). 
> - Outcome: Sufficient evidence to reject our null hypothesis

> - **Hypothesis 2 -** 
> - alpha = .025
> - $H_0$: Is monthly payment greater for those who churned and than those who have not churn $\mu_{monthly payment (churned)} > \mu_{monthly payment (retained)}$.  
> - $H_a$: Rejection of null $\mu_{monthly payment (churned)} <= \mu_{monthly payment (retained)}$. 
> - Outcome: Sufficent evidence to reject our null hypothesis

> - **Question 3 -** 
> - Which service provided has the highest churn?
> - Outcome: Movie Streaming (and TV streaming close second).

> - **Question 4 -** 
> - When (in terms of tenure) are people most likely to churn?
> - Outcome: Within the first 3 months, the first month being the highest.

<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

> - I found that of the classification models I created, LogisticRegression, DecisionTree, and RandomForest predicted the likelyhood of Churn equally well.
> - I chose my RandomForest model as my best model with an 80% accuracy rate for predicting my target value, Churn. This model outperformed my baseline score of 73% accuracy, so it has a little value.
> - Some initial exploration and statistical testing revealed that engineering some new features around tenure, services, and internet type I might help my models predict with even more accuracy, and with more time, I would like to test this hypothesis.

<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- Ask four questions, two of which are to be statistical  and two are to be visualized with context provided.
- Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train three different classification models.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Create csv file with the customer id, the probability of the target values, and the model's prediction for churn for each observation in my test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire
> - Store functions that are needed to acquire data from the measures and species tables from the telco database on the Codeup data science database server; make sure the acquire.py module contains the necessary imports to run my code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare
> - Store functions needed to prepare the telco data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, species. 
> - Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to species (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
> - Feature Selection (after initial iteration through pipeline): Are there any variables that seem to provide limited to no additional information? If so, remove them.
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Introduce myself and my project goals at the very beginning of my notebook walkthrough.
> - Summarize my findings at the beginning like I would for an Executive Summary. (Don't throw everything out that I learned from Storytelling) .
> - Walk Codeup Data Science Team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px white; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- Read this README.md
- Download the aquire.py, prepare.py, and final_report.ipynb files into your working directory
- Add your own env file to your directory. (user, password, host)
- Run the final_report.ipynb notebook
