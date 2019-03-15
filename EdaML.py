import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

training_data = pd.read_csv('../input/train_V2.csv')

train, pre_train = train_test_split(training_data, test_size=0.20 , random_state=42)

pre_train.shape

sns.jointplot(x="winPlacePerc", y="kills", data=pre_train, height=10, ratio=3, color="r")
plt.show()

sns.jointplot(x="winPlacePerc", y="walkDistance",  data=pre_train, height=10, ratio=3, color="lime")
plt.show()

sns.jointplot(x="winPlacePerc", y="rideDistance", data=pre_train, height=10, ratio=3, color="m")
plt.show()

sns.jointplot(x="winPlacePerc", y="heals", data=pre_train, height=10, ratio=3, color="b")
plt.show()

sns.jointplot(x="winPlacePerc", y="boosts", data=pre_train, height=10, ratio=3, color="c")
plt.show()

f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(pre_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


target = 'winPlacePerc'
colsdrop = ['Id', 'groupId', 'matchId', 'matchType', target]
colsfit = [col for col in pre_train.columns if col not in colsdrop]
train , val = train_test_split(pre_train, test_size=0.05)

from lightgbm import LGBMRegressor
parameters = {
    'n_estimators': 100,
    'learning_rate': 0.3, 
    'num_leaves': 20,
    'objective': 'regression_l2', 
    'metric': 'mae',
    'verbose': -1,
}

model = LGBMRegressor(**parameters)
model.fit(train[colsfit], train[target],
    eval_set=[(val[colsfit], val[target])],
    eval_metric='mae',
    verbose=20,
)

feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, colsfit)), columns=['Value','Feature'])

plt.figure(figsize=(16, 9))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('Important features')
plt.tight_layout()

sns.jointplot(x="damageDealt", y="kills", data=pre_train, height=10, ratio=3, color="r")
plt.show()

z = 8
f,ax = plt.subplots(figsize=(10, 10))
columns = pre_train.corr().nlargest(z, 'winPlacePerc')['winPlacePerc'].index
cmap = np.corrcoef(train[columns].values.T)
sns.set(font_scale=1.25)
hmap = sns.heatmap(cmap, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)
plt.show()