import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('data.csv')

df['alcohol_consumption'] = df['alcohol_consumption'].fillna('None')
df['symptoms_list'] = df['symptoms_list'].fillna('None')

#  Define Feature Groups
numeric_features = [
    'age', 'bmi', 'vitamin_a_percent_rda', 'vitamin_c_percent_rda', 
    'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 'vitamin_b12_percent_rda', 
    'folate_percent_rda', 'calcium_percent_rda', 'iron_percent_rda', 'symptoms_count'
]

# Ordinal features (categories with a specific order)
ordinal_features = ['exercise_level', 'sun_exposure', 'income_level', 'latitude_region']
ordinal_ordering = [
    ['Sedentary', 'Light', 'Moderate', 'Active'], 
    ['Low', 'Moderate', 'High'],                 
    ['Low', 'Middle', 'High'],                   
    ['Low', 'Mid', 'High']                       
]

# Nominal features (categories without order)
nominal_features = ['gender', 'smoking_status', 'alcohol_consumption', 'diet_type']

#  Build the Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('ord', OrdinalEncoder(categories=ordinal_ordering), ordinal_features),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ]
)

#  Apply Transformation
# We drop 'symptoms_list' (text) and the target 'vitamin_deficiency'
X = df.drop(columns=['symptoms_list', 'vitamin_deficiency'])
y = df['vitamin_deficiency']

X_processed = preprocessor.fit_transform(X)

# Convert back to a DataFrame for readability
onehot_cols = preprocessor.named_transformers_['nom'].get_feature_names_out(nominal_features)
all_cols = numeric_features + ordinal_features + list(onehot_cols)
df_processed = pd.DataFrame(X_processed, columns=all_cols)
df_processed['target'] = y.values

# Save the clean data
df_processed.to_csv('preprocessed_data.csv', index=False)
print("Preprocessing complete! Saved to 'preprocessed_data.csv'")