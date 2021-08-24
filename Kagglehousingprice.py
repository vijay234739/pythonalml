from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd_df
import numpy as np
import xgboost as xgb
import sklearn.metrics as mets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


# defining  reading CSV file
def load_analysis_data(v_in_file):
    return pd_df.read_csv(v_in_file)


def split_data(in_df_clean_base, in_file_process_flag):
    if in_file_process_flag == 'Trainset':
        X_train, X_test, y_train, y_test = train_test_split(in_df_clean_base, in_df_clean_base['SalePrice'],
                                                            test_size=0.2, stratify=in_df_clean_base['YearRemodAdd'])
        # X_train = pd_df.DataFrame(X_train, columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
        # X_test = pd_df.DataFrame(X_test, columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
    else:
        X_test = pd_df.DataFrame(in_df_clean_base,
                                 columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
        y_test = pd_df.DataFrame(in_df_clean_base,
                                 columns=['PassengerId', 'Survived'])
        X_train = [0]
        y_train = [0]
    return X_train, X_test, y_train, y_test


# data analysis
v_train_file = 'train_HousePrice.csv'
v_house_base = load_analysis_data(v_train_file)
v_house_corr = v_house_base.corr()

# print(corr.sort_values(corr["SalesPrice"], ascending=False))
highest_corr_features = v_house_corr.index[abs(v_house_corr["SalePrice"]) > 0.5]
Final_columns = highest_corr_features.values
Final_column = np.append(Final_columns, ['Id'])
#print(Final_column)
v_house_reduce = pd_df.DataFrame(v_house_base, columns=Final_column)
Final_column_val = np.delete(Final_column, np.where(Final_column == 'SalePrice'))

# print(v_house_reduce["SalePrice"].skew())

# std_scale = StandardScaler()
# X_s = std_scale.fit_transform(v_house_reduce.columns)
# v_house_std = pd_df.DataFrame(X_s)   # Put the np array back into a pandas DataFrame for later
# print(v_house_std.info())

X_train, X_test, y_train, y_test = split_data(v_house_reduce, 'Trainset')

X_train = pd_df.DataFrame(X_train, columns=Final_column_val)
X_test  = pd_df.DataFrame(X_test, columns=Final_column_val)

xgb_boost = xgb.XGBRegressor(eta=0.2, max_depth=5,subsample=0.8)
xgb_model = xgb_boost.fit(X_train, y_train, eval_metric=mets.r2_score)

reg1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2)
reg2 = RandomForestRegressor(n_estimators=100, max_depth=4)
reg3 = LinearRegression()

reg1.fit(X_train, y_train)
reg2.fit(X_train, y_train)
reg3.fit(X_train, y_train)

ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg.fit(X_train, y_train)

xt = X_test

pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = ereg.predict(xt)
pred5 = xgb_model.predict(xt)

plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='LinearRegression')
plt.plot(pred5, 'xg', label='Xgboost')
plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')

plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')

#plt.show()

v_train_file = 'testHousePrice.csv'
v_house_base_test = load_analysis_data(v_train_file)

#print(Final_column_val)
v_house_base_test_red = pd_df.DataFrame(v_house_base_test, columns=Final_column_val)
print(v_house_base_test_red.info())


median_impute = SimpleImputer(strategy="median")
median_impute.fit(v_house_base_test_red)
X_val = median_impute.transform(v_house_base_test_red)
v_in_df_data_clean = pd_df.DataFrame(X_val, columns=v_house_base_test_red.columns)
print(v_in_df_data_clean.info())


pred6 = ereg.predict(v_in_df_data_clean)
pred7 = xgb_model.predict(v_in_df_data_clean)

## Save Result in Output File
submit_train = pd_df.DataFrame({"PassengerId": X_test.Id, 'calculated_voting': pred4, 'Actual': y_test,
                           'xgboost': pred5, 'Score_voting': mets.r2_score(pred4, y_test),
                           'Score_voting_xgb': mets.r2_score(pred5, y_test)})

submit = pd_df.DataFrame({"Id": v_house_base_test_red.Id, 'SalePrice': pred6, 'SalePricexgb': pred7})
#submit = pd_df.DataFrame(v_in_df_data_clean)
submit.to_csv("Submission_houseprice_28.csv", index=False)
submit_train.to_csv("Submission_houseprice_test1.csv", index=False)
