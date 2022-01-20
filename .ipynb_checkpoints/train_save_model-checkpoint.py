# model's imports
import pandas as pd
import numpy as np
import math
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#_____________________________________________________________________________________________
#Functions for loading the input dataset
def load_portfolio(portfolio_file_path: str) -> pd.DataFrame:
    '''Function for loading JSON files with file path.'''
    try:
        portfolio = pd.read_json(portfolio_file_path, 
                                 orient='records', 
                                 lines=True)
        return portfolio 
    except FileNotFoundError:
        print("WARNING: File path doesn't exist")

def load_profile(profile_file_path: str) -> pd.DataFrame:
    '''Function for loading CSV files with file path.'''
    try:
        profile = pd.read_json(profile_file_path,
                                orient='records', 
                                 lines=True)

        return profile
    except FileNotFoundError:
        print("WARNING: File path doesn't exist")

def load_transcript(transcript_file_path: str) -> pd.DataFrame:
    '''Function for loading CSV files with file path.'''
    try:
        transcript = pd.read_json(transcript_file_path,
                                   orient='records', 
                                    lines=True)
        return transcript
    except FileNotFoundError:
        print("WARNING: File path doesn't exist")

#_____________________________________________________________________________________________
#Functions for cleaning the datasets
def portfolio_cleaning(portfolio: pd.DataFrame) -> pd.DataFrame:
    '''Function for cleaning portfolio dataframe.
    Parameters
    ----------
    dataset : pd.DataFrame
        portfolio dataframe 

    Returns
    -------
    pd.DataFrame
         clean portfolio dataframe
    '''
    # Change the unit of 'duration' column from days to hours
    portfolio['duration'] = portfolio['duration']*24
    portfolio.rename({'id': 'offer_id','duration':'duration_h'}, axis=1, inplace= True)
    
    # Apply one hot encoding to channels column
    portfolio['web'] = portfolio['channels'].apply(lambda x: 1 if 'web' in x else 0)
    portfolio['email'] = portfolio['channels'].apply(lambda x: 1 if 'email' in x else 0)
    portfolio['mobile'] = portfolio['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
    portfolio['social'] = portfolio['channels'].apply(lambda x: 1 if 'social' in x else 0)
    
    # Drop channels column
    portfolio.drop(['channels'], axis=1, inplace=True)
    
    # Replace categorical variable to numeric in offer_type 
    portfolio['offer_type'].replace(['bogo', 'informational','discount'],[0, 1,2], inplace=True)
    return portfolio
    
def profile_cleaning(profile:pd.DataFrame) -> pd.DataFrame:
    '''Function for cleaning profile dataframe.
    Parameters
    ----------
    dataset : pd.DataFrame
        profile dataframe 

    Returns
    -------
    pd.DataFrame
         clean profile dataframe
    '''
    # Drop all rows with 118 age  
    profile.drop(profile[profile['age'] == 118].index, inplace = True)
    
    # rename id column to customer_id
    profile.rename({'id': 'customer_id'}, axis=1, inplace= True)
    
    # Change became_member_on type to the right one
    profile['became_member_on'] = profile['became_member_on'].astype(str).astype('datetime64[ns]')
    
    # Replace categorical variable with numeric in gender 
    profile['gender'].replace(['O', 'M','F'],
                        [0, 1,2], inplace=True)
    
    
    return profile

def transcript_cleaning(transcript:pd.DataFrame) -> pd.DataFrame:
    '''Function for cleaning transcript dataframe.
    Parameters
    ----------
    dataset : pd.DataFrame
        transcript dataframe 

    Returns
    -------
    pd.DataFrame
         clean transcript dataframe
    '''
    # Rename person to customer_id
    transcript.rename({'person': 'customer_id'}, axis=1, inplace= True)
    
    # Remove customer id's that are not in the customer profile DataFrame
    transcript = transcript[transcript['customer_id'].isin(profile['customer_id'])]
    transcript = pd.concat([transcript, transcript['value'].apply(pd.Series)], axis=1)
    # Clean up the duplicates in offer id and offer_id and meger into one column
    transcript['clean_id'] = np.where(transcript['offer id'].isnull() &
                                      transcript['offer_id'].notnull(), transcript['offer_id'],transcript['offer id'])

    # Drop the original id columns
    transcript.drop(['offer id', 'offer_id'], axis=1, inplace=True)

    # Rename the offer_id column
    transcript.rename(columns={'clean_id': 'offer_id'}, inplace=True)
    
    # Drop value column
    transcript.drop('value', axis=1, inplace=True)

    # Drop amount and reward columns since they have a huage number of missing valuse
    transcript.drop('amount', axis=1, inplace=True)
    transcript.drop('reward', axis=1, inplace=True)

    # Drop transaction rows
    transcript.drop(transcript[transcript['event']== 'transaction'].index,axis=0,inplace = True)
    
    # Rename time to time_h to represent the time measurement
    transcript.rename({'time':'time_h'}, axis=1, inplace=True)
    
    # Drop offer received rows
    transcript.drop(transcript[transcript['event']=='offer received'].index, inplace=True,axis=0)
    
    # Rename time to time_h to represent the time measurement
    transcript.rename({'time':'time_h'}, axis=1, inplace=True)
    
    # Replace categorical variable with numeric in event 
    transcript['event'] = transcript['event'].apply(lambda x: 0 if 'offer viewed' in x else 1)
    
    return transcript


#_____________________________________________________________________________________________
#Function for merging the three clean datasets
def combined_datasets(portfolio:pd.DataFrame, profile:pd.DataFrame, transcript:pd.DataFrame) -> pd.DataFrame:
    """Function for combining the cleaned dataframes

    Parameters
    ----------
    portfolio: pd.DataFrame
        cleaned dataframe 
    profile: pd.DataFrame
        cleaned dataframe 
    transcript: pd.DataFrame
        cleaned dataframe     
    cfg: dict
        dictionary with configuration loaded from file

    Returns
    -------
    pd.DataFrame
         The combined dataframe
    """
    master_df = transcript.merge(portfolio,how='left',on='offer_id')
    master_df = master_df.merge(profile,how='left', on ='customer_id')

    return master_df
    


#_____________________________________________________________________________________________
#Function for training the model
def model_training(master_df: pd.DataFrame, cfg: dict) -> list:
    """Function for training the loaded dataframe

    Parameters
    ----------
    master_df : pd.DataFrame
        input dataframe with all data
    cfg: dict
        dictionary with configuration loaded from file

    Returns
    -------
    class
         sklearn.tree._classes.RandomForestClassifier
    """
    global dt
    # Select the columns/features from the Pandas dataframe that we want to use in the model:
    features = cfg['COLS']['FEATURES_TO_USE']
    target = cfg['COLS']['TARGET']

    X = np.array(master_df[features])
    y = np.array(master_df[target]).ravel()
    
    # Normalizing some numerical values 
    #scaler = MinMaxScaler()
    #features_scaled = ['time_h','duration_h', 'difficulty']
    #X_scaled = X.copy()
    #X_scaled[features_scaled] = scaler.fit_transform(X_scaled[features_scaled])
    
    # Train and split the dataframe
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= cfg['MODEL']['TRAIN_SIZE'],
                                                        random_state= cfg['MODEL']['RANDOM_STATE'] )

    # Create a Random Forest Classifier model that we can train
    dt = DecisionTreeClassifier()
    
    # Train Random Forest Classifier model
    dt.fit(X_train, y_train)
    
    return dt

#_____________________________________________________________________________________________
#Function for saving the model
def save_model_to_pickle(trained_model):
    with open("Os_model.pkl", "wb") as my_stream:
        pickle.dump(trained_model, my_stream)

#_____________________________________________________________________________________________
#Main function to run ML model
if __name__ == "__main__":
    """Function for excuating all functions"""
    cfg = json.load(open("config.json"))
    
    # Assgining the paths
    portfolio_path = cfg["DATA"]["PORTFOLIO_PATH"]
    profile_path = cfg["DATA"]["PROFILE_PATH"]
    transcript_path = cfg["DATA"]["TRANSCRIPT_PATH"]
    
    # Loading the datasets
    portfolio = load_portfolio(portfolio_path)
    profile = load_profile(profile_path)
    transcript = load_transcript(transcript_path)
    
    # Cleaning the dataframes
    Cleaned_portfolio = portfolio_cleaning(portfolio)
    Cleaned_profile = profile_cleaning(profile)
    Cleaned_transcript = transcript_cleaning(transcript)
    
    # Combining the dataframes
    master_df= combined_datasets(Cleaned_portfolio, Cleaned_profile, Cleaned_transcript)
    
    # Training the model
    trained_model = model_training(master_df, cfg)
    
    # Load the model into variable
    Os_trained_model = trained_model
    
    #save model to disk
    save_model_to_pickle(Os_trained_model)