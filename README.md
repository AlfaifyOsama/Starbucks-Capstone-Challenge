![Predictive-Maintenance-Model](/Picture1.jpg)
# Starbucks Capstone Challenge
Here is the Medium blog post I have written: 

## 1. The problem


## 2. Libraries

I use Python3. Here are the libraries I used in my Jupyter Notebook:
- Numpy
- Pandas
- Sklearn
- Seaborn
- matplotlib.pyplot

## 3. The repository files:
- Data File: the datasets that I used in this project
- Starbucks_Capstone_notebook: to explore the datasets
- Final API_test: API test to show the preduction
- config.json: to set the parameters for the model
- train_save_model.py: excutable python code to do the required analysis and modeling

## 4. Summary of Results
Through this project, I analyzed the provided dataset by Starbucks and then build a model that can predict whether a customer would complete or view the offer?

First, I explored each dataset, visualize it to get an overall understanding on the data. Then, I moved to the Preprocessing Part, which took most of the time and effort. After that, I tested different models to see which is best for this case. 

I got the following insights from the datasets:

- According to the available data, There are three ‘gender’ categories into which the customers falls in ( M, F and O).
- Male Customers (8484 men) are more than Female Customers(6129 women) with 57% of customers are Males compared to 41% Females. 

- There are 2175 missing values. 
    
- There are 212 customers chose “O” as their gender.

- The most common offer type among all age groups is the BOGO , followed by the Discount Offers. Whereas, the least common offer to be sent is the informational offers. 
## 5. How to run the API
You need to run the following files in the terminal:
    1- `python train_save_model.py`
    2- `python model_API.py`
    3- `uvicorn model_API:app --port=5000` 
## 5. Acknowledgments
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org)
- [matplotlib](https://matplotlib.org)
- [GitHub Docs](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
