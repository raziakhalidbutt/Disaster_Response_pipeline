# Disaster Response Pipeline Project

# Table of content
1. [Installation](#Installation)
2. [Instructions](#Instructions)
3. [Motivation](#Motivation)
4. [File Description](#FileDescription)
5. [Results](#Results)
6. [Acknowledgements](#Acknowledgements)

<a name="Installation"></a>
# Installation
You need Anaconda distribution of python 3.* version. The additional libraries required for this project are:

1. nltk
2. sqlalchemy

<a name="Instructions"></a>
# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="Motivation"></a>
# Motivation
This project is part of Data Scientist Nanodegree Program from Udacity. The goal is to run ETL (Extract Transform Load) pipline and ML (Machine learning) pipeline to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

<a name="FileDescription"></a>
# File Description

There two dataset files provided by [Figure Fight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) are in the data folder.

    disaster_categories.csv: Categories of the messages
    disaster_messages.csv: Multilingual disaster response messages

The project is done in three steps that includes:

1. ETL Pipeline: In data folder

**process_data.py:** The process include for data cleaning pipeline.

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline: In models foder

**train_classifier.py:** The process for a machine learning pipeline includes:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App: In app folder

**runner.py:** To run web app
    
<a name="Results"></a>
# Results
Following are the web app screenshots:

1. Enter messages in the text box and click classify that highlights the categories message belong to.

![images](https://github.com/raziakhalidbutt/Disaster_Response_pipeline/blob/master/images/fig1.PNG)

![images](https://github.com/raziakhalidbutt/Disaster_Response_pipeline/blob/master/images/fig4.PNG)

2. The graphs of training dataset provided by Figure Fight.

![images](https://github.com/raziakhalidbutt/Disaster_Response_pipeline/blob/master/images/fig2.PNG)

![images](https://github.com/raziakhalidbutt/Disaster_Response_pipeline/blob/master/images/fig3.PNG)


<a name="Acknowledgements"></a>
# Acknowledgements
This dataset is provided by Figure Fight. 
