from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pprint
import pandas as pd

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

# Encode string values
replace_map = {'thal': {'normal': 0, 'reversible_defect': 1, 'fixed_defect': 2, }}
train_X_replace = train_X.copy()
train_X_replace.replace(replace_map, inplace=True)
test_X_replace = test_X.copy()
test_X_replace.replace(replace_map, inplace=True)

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1) # CHANGE MODEL HERE
rf_model.fit(train_X_replace, train_Y) # Train model
rf_val_predictions = rf_model.predict(test_X_replace) # Predict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_x, val_x, train_y, val_y = train_test_split(train_X_replace, train_Y, random_state=1, test_size = 0.2)
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_x, train_y)
rf_val_preds = rf_model.predict(val_x)
rf_val_mae = mean_absolute_error(rf_val_preds, val_y)

print(rf_val_mae)