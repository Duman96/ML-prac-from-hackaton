from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pprint

# Path of the file to read.
heart_train_features_file_path = 'train_values.csv'
heart_train_labels_file_path = 'train_labels.csv'
heart_test_features_file_path = 'test_values.csv'

heart_train_features = pd.read_csv(heart_train_features_file_path)
heart_train_labels = pd.read_csv(heart_train_labels_file_path)
heart_test_features = pd.read_csv(heart_test_features_file_path)
# Create train and test without patient ID 
features = ['slope_of_peak_exercise_st_segment', 'thal', 'resting_blood_pressure',
            'chest_pain_type', 'num_major_vessels', 'fasting_blood_sugar_gt_120_mg_per_dl', 
            'resting_ekg_results', 'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex',
            'age', 'max_heart_rate_achieved', 'exercise_induced_angina']
train_X = heart_train_features[features]
train_Y = heart_train_labels.heart_disease_present
test_X = heart_test_features[features]
replace_map = {'thal': {'normal': 0, 'reversible_defect': 1, 'fixed_defect': 2, }}
train_X_replace = train_X.copy()
train_X_replace.replace(replace_map, inplace=True)
train_X_replace.head()
test_X_replace = test_X.copy()
test_X_replace.replace(replace_map, inplace=True)
test_X_replace.head()

train_x, val_x, train_y, val_y = train_test_split(train_X_replace.as_matrix(), train_Y.as_matrix(), test_size=0.1)
# Define the model. Set random_state to 1

my_imputer = Imputer()
train_x = my_imputer.fit_transform(train_x)
val_x = my_imputer.transform(val_x)

rf_model = LogisticRegression()
rf_model.fit(train_x, train_y)
rf_val_preds = rf_model.predict_proba(val_x)[:,1]
rf_val_mae = mean_absolute_error(rf_val_preds, val_y)

print(rf_val_preds)
print(val_y)
print(rf_val_mae)
