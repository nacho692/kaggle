import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import shap
import matplotlib.pyplot as plt
import inspect
from scikeras.wrappers import KerasClassifier
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense
from tensorflow.python.keras import regularizers, callbacks
from itertools import combinations

def get_keras_model(type, cols):
    if type == 'estandar':
        return estandar(cols)
    elif type == 'deep':
        return deep(cols)
def estandar(cols):
    # create model
    model = Sequential()
    model.add(InputLayer((cols,)))
    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def deep(cols):
    # create model
    model = Sequential()
    model.add(InputLayer((cols,)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_KerasClassifier_for_grid_search(cols):
    es = callbacks.EarlyStopping()
    clf = KerasClassifier(verbose=3, callbacks=[es])
    return "Keras", clf, {'model':[deep(cols), estandar(cols)], 'epochs': [5], 'batch_size': [100]}

def get_input_output(df, input_cols=None, output_cols=None, remove_cols=None):
    if remove_cols is None:
        remove_cols = ['id']
    if output_cols is None:
        output_cols = ['smoking']
    if input_cols is None:
        input_cols = [c for c in df.columns if c not in remove_cols and c not in output_cols]
    X = df[input_cols].values
    y = df[output_cols].values
    shape = (y.shape[0],)
    data = {
        'X': X,
        'y': y.reshape(shape),
        'columns': input_cols
    }
    return data

def scaler_preprocess(X):
    scaler = StandardScaler().fit(X)
    X_processed = scaler.transform(X)

    data = {
        'X': X_processed,
    }
    return data

def PCA_preprocess(X):
    X_PCA = PCA().fit_transform(X)
    X_processed = np.concatenate([X, X_PCA], axis=1)
    data = {
        'X': X_processed,
    }
    return data

def preprocess_eyesight(df):

    cols = ['eyesight(left)', 'eyesight(right)']
    df.loc[:, 'best_eyesight_is_left'] = False
    df.loc[:, 'best_eyesight_is_right'] = False

    df.loc[df['eyesight(left)'] > df['eyesight(right)'], 'best_eyesight_is_left'] = True
    df.loc[df['eyesight(left)'] < df['eyesight(right)'], 'best_eyesight_is_right'] = True
    # values = df[cols].values.flatten()
    # thermos = generate_thermos(values)
    # for key, value in thermos:
    #     df.loc[:, f'eyesight(left)_{key}'] = False
    #     df.loc[:, f'eyesight(right)_{key}'] = False
    #     df.loc[df['eyesight(left)'] >= value, f'eyesight(left)_{key}'] = True
    #     df.loc[df['eyesight(right)'] >= value, f'eyesight(right)_{key}'] = True

    df.drop(cols, axis=1, inplace=True)
    df.fillna(True, inplace=True)

    data = {'df': df}
    return data

def preprocess_hearing(df):

    df.loc[:, 'best_hearing_is_left'] = df['hearing(left)'] > df['hearing(right)']
    df.loc[:, 'best_hearing_is_right'] = df['hearing(left)'] < df['hearing(right)']
    df.loc[:, 'hearing(left)'] = df.loc[:, 'hearing(left)'] - 1
    df.loc[:, 'hearing(right)'] = df.loc[:, 'hearing(right)'] - 1

    data = {'df':df}

    return data

def compute_pairwise_coefficients(df, columns_to_take_into_consideration=None):
    if columns_to_take_into_consideration is None:
        columns_to_take_into_consideration = ['weight(kg)', 'waist(cm)', 'systolic', 'age', 'height(cm)',
        'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
        'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST', 'ALT', 'Gtp']
    extra_columns = []
    for col_1, col_2 in combinations(columns_to_take_into_consideration, 2):
        column_key = f"{col_1}_{col_2}"
        df[column_key] = df[col_1] / df[col_2]
        extra_columns.append(column_key)
    data = {
        'df':df,
        'thermos_columns': columns_to_take_into_consideration+extra_columns
    }
    return data
def preprocess_thermos(df, thermos_columns=None):
    if thermos_columns is None:
        thermos_columns = ['weight(kg)', 'waist(cm)', 'systolic', 'age', 'height(cm)',
        'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
        'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST', 'ALT', 'Gtp']
    for col in thermos_columns:
        values = df[col].values.flatten()
        thermos = generate_thermos(values)
        for key, value in thermos:
            df.loc[:, f'{col}_{key}'] = False
            df.loc[df[col] >= value, f'{col}_{key}'] = True
    df.drop(thermos_columns, axis=1, inplace=True)

    data = {
        'df':df
    }
    return data

def preprocess_one_hot(df, one_hot_columns=None):
    if one_hot_columns is None:
        one_hot_columns = ['Urine protein']
    for col in one_hot_columns:
        values = df[col].unique()
        thermos = generate_one_hot(values)
        for key, value in thermos:
            df.loc[:, f'{col}_{key}'] = df[col] == value
    df.drop(one_hot_columns, axis=1, inplace=True)
    data = {
        'df':df
    }
    return data
def generate_thermos(values):
    percentiles = set(np.percentile(values, [i for i in range(0,100, 10)]))
    percentiles = [a for a in enumerate(percentiles)]

    percentiles.sort(key=lambda x: x[1])
    return percentiles

def generate_one_hot(values):
    one_hot_values = set(values)
    one_hot_values = [a for a in enumerate(one_hot_values)]
    return one_hot_values
def preprocess(df, preprocessing_functions):
    input_data = {'df': df}

    for f in preprocessing_functions:
        args = inspect.getfullargspec(f).args
        sub_input = {k: input_data[k] for k in args if k in input_data}
        res = f(**sub_input)
        input_data.update(res)

    return input_data['X'], input_data['y'], input_data['columns']

def get_MLPClassifier_for_grid_search():
    alphas = [0.1, 0.15, 0.25]
    layers = [(300, 100, i) for i in range(10, 60, 10)]

    parameters = {'alpha': alphas, 'hidden_layer_sizes': layers}
    clf = MLPClassifier()
    return 'MLPC', clf, parameters

def get_KerasClassifier_for_grid_search(cols):
    es = callbacks.EarlyStopping(monitor="loss")
    clf = KerasClassifier(verbose=3, callbacks=[es])
    return "Keras", clf, {'epochs': [25, 50], 'batch_size': [32], 'model': [lambda : get_keras_model('deep', cols), lambda: get_keras_model('estandar',cols)]}
def print_grid_search_results(results):
    params_with_values = [(p, m, s) for p, m, s in
                          zip(results['params'], results['mean_test_AUC'], results['std_test_AUC'])]
    best_result = np.argmin(results['rank_test_AUC'])
    winner_params = params_with_values[best_result]
    winner_params_str = " ".join([f"{key}: {value}" for key, value in winner_params[0].items()])
    print(f'Winner is [{winner_params_str}] {winner_params[1]:.4f} mean. {winner_params[2]:.2f} std')
    for params, mean, std in params_with_values:
        params_str = " ".join([f"{key}: {value}" for key, value in params.items()])
        print(f"[{params_str}] {mean:.4f} mean. {std:.2f} std")

def get_shapley_values(classifier, X, y, columns, amount_of_samples=400):
    indxs = np.choice([i for i in range(X.shape[0])], amount_of_samples)
    X_samples = X[indxs]
    classifier.fit(X, y)
    y_pred = clf.predict(X)
    false_positives = [i for i in range(len(y)) if y[i] != y_pred[i] and y[i] == 0]
    false_negatives = [i for i in range(len(y)) if y[i] != y_pred[i] and y[i] == 1]

    explainer = shap.Explainer(clf.predict, X_samples, feature_names=columns)
    shap_values_positives = explainer(X[false_positives])
    shap_values_negatives = explainer(X[false_negatives])

    shap.plots.beeswarm(shap_values_negatives)
    plt.plot()
    shap.plots.beeswarm(shap_values_positives)
    plt.plot()

if __name__=='__main__':
    df = pd.read_csv('../../data/smoking/train.csv')
    for col in df.columns:
        if col not in ('id', 'smoking'):
            print(col)
            print(df[col].nunique())
            print('='*4)

    preprocessing_functions = [
        compute_pairwise_coefficients,
        preprocess_eyesight,
        preprocess_hearing,
        preprocess_thermos,
        preprocess_one_hot,
        get_input_output,
        scaler_preprocess
    ]
    X, y, cols = preprocess(df, preprocessing_functions)
    pdb.set_trace()
    classifiers_to_explore = [
        #get_MLPClassifier_for_grid_search()
        get_KerasClassifier_for_grid_search(len(cols))
    ]
    for name, clf, parameters in classifiers_to_explore:
        gs = GridSearchCV(clf, parameters, n_jobs=1, scoring={"AUC":"roc_auc"}, refit="AUC", verbose=3,  return_train_score=True)

        gs.fit(X, y)
        results = gs.cv_results_
        print(name)
        print_grid_search_results(results)