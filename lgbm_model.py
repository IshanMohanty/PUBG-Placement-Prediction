'''
@author: Ishan Mohanty
EE660: Machine Learning From Signals
'''

#dependencies
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc, sys
gc.enable()

# Thanks and credited to https://www.kbindle.com/gemartin who created this wonderful mem reducer
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#expanding feature space with highly correlated features
 def feature_expansion(df):
    
    print("Starting feature expansion")
    
    df['headShotRate'] = df['kills']/df['headshotKills']
    df['killStreakRate'] = df['killStreaks']/df['kills']
    df['health'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]
    df['items'] = df['heals'] + df['boosts']
    df['teamwork'] = df['assists'] + df['revives']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN    
    print("Removing all NaN's from the dataset")
    df.fillna(0, inplace=True)
    
    print("Feature expansion completed")


#process features
def processing_features(df,is_train=True):
    
    if is_train:
        test_idx = None
    else:
        test_idx = df.Id
        
    print("deleting specific columns")
    response = 'winPlacePerc'
    
    feature_list = list(df.columns)
    feature_list.remove("Id")
    feature_list.remove("matchId")
    feature_list.remove("groupId")
    features.remove("matchType")
     
    y_label = None
       
    if is_train: 
        print("retrieve response")
        y_label = np.array(df.groupby(['matchId','groupId'])[response].bind('mean'), dtype=np.float64)
        feature_list.remove(response)

    print("retrieve group mean feature")
    bind = df.groupby(['matchId','groupId'])[feature_list].bind('mean')
    bind_rank = bind.groupby('matchId')[feature_list].rank(pct=True).reset_index()
    
    if is_train: df_modified = bind.reset_index()[['matchId','groupId']]
    else: df_modified = df[['matchId','groupId']]

    df_modified = df_modified.merge(bind.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_modified = df_modified.merge(bind_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    print("retrieve group max feature")
    bind = df.groupby(['matchId','groupId'])[feature_list].bind('max')
    bind_rank = bind.groupby('matchId')[feature_list].rank(pct=True).reset_index()
    df_modified = df_modified.merge(bind.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_modified = df_modified.merge(bind_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("retreive group min feature")
    bind = df.groupby(['matchId','groupId'])[feature_list].bind('min')
    bind_rank = bind.groupby('matchId')[feature_list].rank(pct=True).reset_index()
    df_modified = df_modified.merge(bind.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_modified = df_modified.merge(bind_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("retrieve group size feature")
    bind = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_modified = df_modified.merge(bind, how='left', on=['matchId', 'groupId'])
    
    print("retrieve match mean feature")
    bind = df.groupby(['matchId'])[features].bind('mean').reset_index()
    df_modified = df_modified.merge(bind, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    print("retrieve match size feature")
    bind = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_modified = df_modified.merge(bind, how='left', on=['matchId'])
    
    df_modified.drop(["matchId", "groupId"], axis=1, inplace=True)

    data_x = df_modified
    
    feature_list_names = list(df_modified.columns)

    del df, df_modified, bind, bind_rank
    gc.collect()

    return data_x, y_label, feature_list_names, test_idx



#data and feature pipeline for train
df_train = pd.read_csv('../input/train_V2.csv')  
df_train = df_train[df_train['maxPlace'] > 1]
feature_expansion(df_train)
x_train, y_train, train_columns, _ = processing_features(df_train,True)
x_train = reduce_mem_usage(x_train)
gc.collect()

#data and feature pipeline for test
df_test = pd.read_csv('../input/test_V2.csv')
feature_expansion(df_test)
x_test, _, _ , test_idx = processing_features(df_test,False)
x_test = reduce_mem_usage(x_test)
gc.collect()

#LGBM model run
idx_train = round(int(x_train.shape[0]*0.8))
sample_train_x = x_train[:idx_train] 
sample_valid_x = x_train[idx_train:]
sample_train_y = y_train[:idx_train] 
sample_valid_y = y_train[idx_train:] 
gc.collect();

def train_algo(training_X, training_y, valid_X, valid_y, x_test):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators': 18000, 'early_stopping_rounds': 180,
              "num_leaves" : 21, "learning_rate" : 0.05, "bbinding_fraction" : 0.7,
               "bbinding_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgbm_train = lgb.Dataset(training_X, label=training_y)
    lgbm_val = lgb.Dataset(valid_X, label=valid_y)
    model = lgb.train(params, lgbm_train, valid_sets=[lgbm_train, lgbm_val], early_stopping_rounds=200, verbose_eval=1000)
    
    pred_test = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test, model

prediction, model = train_algo(sample_train_x, sample_train_y , sample_valid_x, sample_valid_y, x_test)


#create submission and do some post processing

submission_df = pd.read_csv("../input/sample_submission_V2.csv")
test_df = pd.read_csv("../input/test_V2.csv")
submission_df['winPlacePerc'] = prediction
# Restore some columns
submission_df = submission_df.merge(test_df[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
submission_df_group = submission_df.groupby(["matchId", "groupId"]).first().reset_index()
submission_df_group["rank"] = submission_df_group.groupby(["matchId"])["winPlacePerc"].rank()
submission_df_group = submission_df_group.merge(
    submission_df_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
submission_df_group["adjusted_perc"] = (submission_df_group["rank"] - 1) / (submission_df_group["numGroups"] - 1)

submission_df = submission_df.merge(submission_df_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
submission_df["winPlacePerc"] = submission_df["adjusted_perc"]

# Deal with edge cases
submission_df.loc[submission_df.maxPlace == 0, "winPlacePerc"] = 0
submission_df.loc[submission_df.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kbindle.com/anycode/simple-nn-baseline-4
subset = submission_df.loc[submission_df.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
submission_df.loc[submission_df.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
submission_df.loc[(submission_df.maxPlace > 1) & (submission_df.numGroups == 1), "winPlacePerc"] = 0
assert submission_df["winPlacePerc"].isnull().sum() == 0

submission_df[["Id", "winPlacePerc"]].to_csv("finalv2_submission.csv", index=False)
