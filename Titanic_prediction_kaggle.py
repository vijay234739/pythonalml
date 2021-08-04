from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd_df
import numpy as np

v_error_tol = 1


# defining  reading CSV file
def load_analysis_data(v_in_file):
    return pd_df.read_csv(v_in_file)


def clean_input_data(v_in_df_data, v_data_type):
    if v_data_type == 'Testset':
        v_in_df_data['Survived'] = v_in_df_data["Sex"].apply(lambda x: 0 if x == 'male' else 0)

    # Column Sex Analysis and Cleaning
    V_Null_Count_Sex = v_in_df_data['Sex'].isnull().sum()
    V_total_Count_Sex = sum(v_in_df_data.value_counts(v_in_df_data['Sex']))
    V_delta_count_Sex = V_total_Count_Sex - V_Null_Count_Sex
    V_delta_count_Sex_percentage = (V_Null_Count_Sex / V_total_Count_Sex) * 100

    # Column Embarked Analysis and Cleaning
    V_Null_Count_Embarked = v_in_df_data['Embarked'].isnull().sum()
    V_total_Count_Embarked = sum(v_in_df_data.value_counts(v_in_df_data['Embarked']))
    #print(V_total_Count_Embarked)
    # V_delta_count_Embarked = V_total_Count_Embarked - V_Null_Count_Embarked
    V_delta_count_Embarked_percentage = (V_Null_Count_Embarked / V_total_Count_Embarked) * 100
    #print(V_delta_count_Embarked_percentage)

    # transforming categorical to Numerical for Sex column
    if V_Null_Count_Sex == 0 or V_delta_count_Sex_percentage < v_error_tol:
        if V_delta_count_Sex_percentage < v_error_tol:
            v_in_df_data = v_in_df_data.dropna(subset=["Sex"])
        v_in_df_data['gender'] = v_in_df_data["Sex"].apply(lambda x: 1 if x == 'male' else 0)
    else:
        print('Please review Data set for column Sex')

    ########## transforming categorical to Embarked
    if V_Null_Count_Embarked == 0 or V_delta_count_Embarked_percentage < v_error_tol:

        if V_delta_count_Embarked_percentage < v_error_tol:
            v_in_df_data = v_in_df_data.dropna(subset=["Embarked"])
            condition_one = (v_in_df_data["Embarked"] == 'S')
            condition_two = (v_in_df_data["Embarked"] == 'C')
            condition_three = (v_in_df_data["Embarked"] == 'Q')
            conditions = [condition_one, condition_two, condition_three]
            choices = [1, 2, 3]
            v_in_df_data["Embarked_val"] = np.select(conditions, choices)
    else:
        print('Please review Data set for column Embarked')

    v_in_df_data_clean = pd_df.DataFrame(v_in_df_data, columns=['PassengerId', 'Survived', 'Pclass', 'gender',
                                                                'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_val'])

    ########Filling Median value for rest of the NA column
    median_impute = SimpleImputer(strategy="median")
    X_val = median_impute.fit_transform(v_in_df_data_clean)
    v_in_df_data_clean = pd_df.DataFrame(X_val, columns=v_in_df_data_clean.columns)
    return v_in_df_data_clean


def split_data(in_df_clean_base, in_file_process_flag):
    if in_file_process_flag == 'Trainset':
        X_train, X_test, y_train, y_test = train_test_split(in_df_clean_base, in_df_clean_base['Survived'],
                                                            test_size=0.2, stratify=in_df_clean_base['Pclass'])
        X_train = pd_df.DataFrame(X_train, columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
        X_test = pd_df.DataFrame(X_test, columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
    else:
        X_test = pd_df.DataFrame(in_df_clean_base,
                                 columns=['PassengerId', 'Pclass', 'gender', 'Age', 'Fare', 'Embarked_val'])
        y_test = pd_df.DataFrame(in_df_clean_base,
                                 columns=['PassengerId', 'Survived'])
        X_train = [0]
        y_train = [0]
    return X_train, X_test, y_train, y_test


def build_model(in_x_train, in_y_train, in_X_test, in_n_estimators, in_max_leaf_nodes):
    # Model Build and Test-- Random Forest Classifier
    rnd_clf = RandomForestClassifier(n_estimators=in_n_estimators, max_leaf_nodes=in_max_leaf_nodes, n_jobs=-1)
    rnd_clf.fit(in_x_train, in_y_train)
    y_final_rf = rnd_clf.predict(in_X_test)
    return rnd_clf, y_final_rf


def model_metrics(in_y_test, in_y_final_rf):
    # Model Metrics
    print('accuracy_score->' + str(accuracy_score(in_y_test, in_y_final_rf)))
    print('recall_score->' + str(recall_score(in_y_test, in_y_final_rf)))
    print('precision_score->' + str(precision_score(in_y_test, in_y_final_rf)))


########reading file for train data
V_file = 'train - Titanic.csv'
V_file_process_flag = 'Trainset'
V_n_estimators = 500
V_max_leaf_nodes = 16
titanic_base = load_analysis_data(V_file)
titanic_clean_base = clean_input_data(titanic_base, V_file_process_flag)
V_X_train, V_X_test, V_y_train, V_y_test = split_data(titanic_clean_base, V_file_process_flag)
# build model and test accuracy of the model
rnd_clf_model, y_final_test_rf = build_model(V_X_train, V_y_train, V_X_test, V_n_estimators, V_max_leaf_nodes)
model_metrics(V_y_test, y_final_test_rf)

## testing of model
V_file = 'test - Titanic.csv'
V_file_process_flag = 'Testset'
V_n_estimators = 500
V_max_leaf_nodes = 16
titanic_base = load_analysis_data(V_file)
titanic_clean_base = clean_input_data(titanic_base, V_file_process_flag)
V_X_train, V_X_test, V_y_train, V_y_test = split_data(titanic_clean_base, V_file_process_flag)
y_final_value = rnd_clf_model.predict(V_X_test)
#print(y_final_value.to_csv)

submit = pd_df.DataFrame({"PassengerId": V_X_test.PassengerId, 'Survived': y_final_value})
submit.to_csv("Titanic_final_submission.csv", index=False)