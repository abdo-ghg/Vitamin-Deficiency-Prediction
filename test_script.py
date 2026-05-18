import os
import pandas as pd
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Classification
def Preprocessing_c(path):
    # read csv('c') done
    # binning done
    # handling nulls (new) done,
    # enconding Sym done
    # encoding features done
    # scaling done
    # feature selection  -> list names final features df[] done
    df= pd.read_csv(path)
    print(df.head())
# /////////////////////////////////////////////////
    bmi_edges = [0, 18.5, 24.9, 29.9, 100] 
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_edges, labels=bmi_labels)

    age_edges = [0, 30, 50, 120]
    age_labels = ['Youth', 'Adults', 'Seniors']
    df['age_group'] = pd.cut(df['age'], bins=age_edges, labels=age_labels)

    df.drop(["age", "bmi"], axis=1, inplace=True)
# //////////////////////////////////////////////////
    numerical_features = ['symptoms_count', 'iron_percent_rda', 'calcium_percent_rda', 'folate_percent_rda',
                      'vitamin_b12_percent_rda', 'vitamin_e_percent_rda', 'vitamin_d_percent_rda',
                      'vitamin_c_percent_rda', 'vitamin_a_percent_rda']
    with open('medians.pkl', 'rb') as f:
        training_medians = pickle.load(f) 
    df[numerical_features] = df[numerical_features].fillna(training_medians)
# //////////////////////////////////////////////
    categorical_features = ['alcohol_consumption', 'exercise_level', 'sun_exposure',
                            'income_level', 'latitude_region', 'bmi_category', 'age_group',
                            'gender', 'symptoms_list', 'diet_type', 'smoking_status']
    df[categorical_features] = df[categorical_features].astype(object).fillna("Missing").astype(str)
    with open(os.path.join(BASE_DIR, 'symptoms_list_c.pkl'), 'rb') as f:
        training_symptoms = pickle.load(f)
    OE, NE, S, _, _, _, _, _, _ = read_pickles()
    features_to_scale = ['symptoms_count', 'iron_percent_rda', 'calcium_percent_rda', 'folate_percent_rda',
                         'vitamin_b12_percent_rda', 'vitamin_e_percent_rda', 'vitamin_d_percent_rda',
                         'vitamin_c_percent_rda', 'vitamin_a_percent_rda']
    df[features_to_scale] = S.transform(df[features_to_scale])
    df['symptoms_list'] = df['symptoms_list'].fillna('')
    for symptom in training_symptoms:
        df[symptom] = df['symptoms_list'].apply(
            lambda x: 1 if symptom in [s.strip() for s in x.split(';')] else 0
        )
    df = df.drop('symptoms_list', axis=1)

   
    ordinal_cols = ['alcohol_consumption', 'exercise_level', 'sun_exposure',
                'income_level', 'latitude_region', 'bmi_category', 'age_group']
    df[ordinal_cols] = OE.transform(df[ordinal_cols])

    nominal_features = ['gender', 'diet_type', 'smoking_status']
    encoded_nominal = NE.transform(df[nominal_features])
    nominal_df = pd.DataFrame(
        encoded_nominal, 
        columns=NE.get_feature_names_out(nominal_features), 
        index=df.index
    )
    df = pd.concat([df.drop(nominal_features, axis=1), nominal_df], axis=1)

    features_selected=['sun_exposure', 'vitamin_a_percent_rda', 'vitamin_c_percent_rda', 'vitamin_d_percent_rda', 'vitamin_b12_percent_rda', 'folate_percent_rda', 'symptoms_count', 'fatigue', 'numbness_tingling']
    return df[features_selected]     

def read_pickles():
    with open(os.path.join(BASE_DIR, 'ordinal_encoder_features.pkl'), 'rb') as f: ordinal_encoder_features = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'nominal_encoder.pkl'), 'rb') as f: nominal_encoder = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'classification_scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
    selected_features = None
    with open(os.path.join(BASE_DIR, 'target_encoder.pkl'), 'rb') as f: target_encoder = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'svc.pkl'), 'rb') as f: svc = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'xgb.pkl'), 'rb') as f: xgb = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'lr.pkl'), 'rb') as f: lr = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'rf.pkl'), 'rb') as f: rf = pickle.load(f)
    return ordinal_encoder_features, nominal_encoder, scaler, selected_features, target_encoder, rf, svc, xgb, lr

def run_models(path):
    # y_pred = predict(x)
    # accuracy = y y-pred
    # print
    _, _, _, _, target_encoder, rf, svc, xgb, lr = read_pickles()
    df = Preprocessing_c(path)
    dff=pd.read_csv(path)
    X = df
    y = target_encoder.transform(dff[['disease_diagnosis']]).ravel()
    rf_pred = rf.predict(X)
    svc_pred = svc.predict(X)
    xgb_pred = xgb.predict(X)
    lr_pred = lr.predict(X)
    rf_pred = target_encoder.inverse_transform(rf_pred.reshape(-1, 1)).ravel()
    svc_pred = target_encoder.inverse_transform(svc_pred.reshape(-1, 1)).ravel()
    xgb_pred = target_encoder.inverse_transform(xgb_pred.reshape(-1, 1)).ravel()
    lr_pred = target_encoder.inverse_transform(lr_pred.reshape(-1, 1)).ravel()
    print("Random Forest Classification Report:")
    print(classification_report(dff['disease_diagnosis'], rf_pred))
    print("SVC Classification Report:")
    print(classification_report(dff['disease_diagnosis'], svc_pred))
    print("XGBoost Classification Report:")
    print(classification_report(dff['disease_diagnosis'], xgb_pred))
    print("Logistic Regression Classification Report:")
    print(classification_report(dff['disease_diagnosis'], lr_pred))
    print("Random Forest Accuracy:", accuracy_score(dff['disease_diagnosis'], rf_pred))
    print("SVC Accuracy:", accuracy_score(dff['disease_diagnosis'], svc_pred))
    print("XGBoost Accuracy:", accuracy_score(dff['disease_diagnosis'], xgb_pred))
    print("Logistic Regression Accuracy:", accuracy_score(dff['disease_diagnosis'], lr_pred))  

# Regression
def Preprocessing_reg(path):
    # read csv('r')
    # binning
    # enconding Sym  
    # encoding features different nominal and ordinal
    # scaling 
    # feature selection  -> list names final features df[] -> different features
    df= pd.read_csv(path)
# /////////////////////////////////////////////////
    bmi_edges = [0, 18.5, 24.9, 29.9, 100] 
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_edges, labels=bmi_labels)

    age_edges = [0, 30, 50, 120]
    age_labels = ['Youth', 'Adults', 'Seniors']
    df['age_group'] = pd.cut(df['age'], bins=age_edges, labels=age_labels)

    df.drop(["age", "bmi"], axis=1, inplace=True)
# //////////////////////////////////////////////////
    numerical_features = ['symptoms_count', 'iron_percent_rda', 'calcium_percent_rda', 'folate_percent_rda',
                          'vitamin_b12_percent_rda', 'vitamin_e_percent_rda', 'vitamin_d_percent_rda',
                          'vitamin_c_percent_rda', 'vitamin_a_percent_rda']
    with open('medians.pkl', 'rb') as f:
        training_medians = pickle.load(f) 
    df[numerical_features] = df[numerical_features].fillna(training_medians)
# //////////////////////////////////////////////
    categorical_features = ['alcohol_consumption', 'exercise_level', 'sun_exposure',
                            'income_level', 'latitude_region', 'bmi_category', 'age_group',
                            'gender', 'symptoms_list', 'diet_type', 'smoking_status']
    df[categorical_features] = df[categorical_features].astype(object).fillna("Missing").astype(str)
    with open(os.path.join(BASE_DIR, 'symptoms_list_reg.pkl'), 'rb') as f:
        training_symptoms = pickle.load(f)
    df['symptoms_list'] = df['symptoms_list'].fillna('')
    for symptom in training_symptoms:
        df[f'sym_{symptom}'] = df['symptoms_list'].apply(
            lambda x: 1 if symptom in [s.strip() for s in x.split(';')] else 0
        )
    df = df.drop('symptoms_list', axis=1)

    OE, _, _, _, _, _, _, _, _ = read_pickles()
    LE,scaler,Linear_Regression,Random_Forest,SVR,XGB = read_pickles_REG()
    ordinal_cols = ['alcohol_consumption', 'exercise_level', 'sun_exposure',
                'income_level', 'latitude_region', 'bmi_category', 'age_group']
    df[ordinal_cols] = OE.transform(df[ordinal_cols])

    Label_cols = ['gender', 'diet_type', 'smoking_status']
    df[Label_cols] = LE.transform(df[Label_cols])
    df[numerical_features] = scaler.transform(df[numerical_features])
    features_selected=[
        'sym_numbness_tingling',
        'sym_bone_pain',
        'sym_muscle_weakness',
        'sym_memory_problems',
        'vitamin_b12_percent_rda',
        'iron_percent_rda',
        'symptoms_count',
        'vitamin_d_percent_rda']
    return df[features_selected]

def read_pickles_REG():
    with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f: label_encoder = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'regression_scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'Linear Regression_model.pkl'), 'rb') as f: Linear_Regression = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'Random Forest_model.pkl'), 'rb') as f: Random_Forest = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'SVR_model.pkl'), 'rb') as f: SVR = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'XGB_model.pkl'), 'rb') as f: XGB = pickle.load(f) 
    return label_encoder, scaler, Linear_Regression, Random_Forest, SVR, XGB


def run_models_reg(path):
    # y_pred = predict(x)
    # r2, mse
    # print
    label_encoder, scaler, Linear_Regression, Random_Forest, SVR, XGB = read_pickles_REG()
    df = Preprocessing_reg(path)
    dff=pd.read_csv(path)
    X = df
    y = scaler.transform(dff[['vitamin_deficiency']]).ravel()
    rf_pred = Random_Forest.predict(X)
    svr_pred = SVR.predict(X)
    xgb_pred = XGB.predict(X)
    lr_pred = Linear_Regression.predict(X)
    rf_pred = scaler.inverse_transform(rf_pred.reshape(-1, 1))
    svr_pred = scaler.inverse_transform(svr_pred.reshape(-1, 1))
    xgb_pred = scaler.inverse_transform(xgb_pred.reshape(-1, 1))
    lr_pred = scaler.inverse_transform(lr_pred.reshape(-1, 1))
    print("Random Forest R2 Score:", r2_score(dff['vitamin_deficiency'], rf_pred))
    print("SVR R2 Score:", r2_score(dff['vitamin_deficiency'], svr_pred))
    print("XGBoost R2 Score:", r2_score(dff['vitamin_deficiency'], xgb_pred))
    print("Logistic Regression R2 Score:", r2_score(dff['vitamin_deficiency'], lr_pred))
    print("Random Forest MSE:", mean_squared_error(dff['vitamin_deficiency'], rf_pred))
    print("SVR MSE:", mean_squared_error(dff['vitamin_deficiency'], svr_pred))
    print("XGBoost MSE:", mean_squared_error(dff['vitamin_deficiency'], xgb_pred))
    print("Logistic Regression MSE:", mean_squared_error(dff['vitamin_deficiency'], lr_pred))



if __name__ == "__main__":
    run_models(r'Data/synthetic_test_data.csv')
    # run_models_reg()
