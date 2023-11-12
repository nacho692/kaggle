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

def preprocess(df, preprocessing_functions):
    input_data = {'df': df}

    for f in preprocessing_functions:
        args = inspect.getfullargspec(f).args
        sub_input = {k: input_data[k] for k in args if k in input_data}
        res = f(**sub_input)
        input_data.update(res)

    return input_data['X'], input_data['y']

def get_MLPClassifier_for_grid_search():
    alphas = [0.1, 0.15, 0.25]
    layers = [(300, 100, i) for i in range(10, 60, 10)]

    parameters = {'alpha': alphas, 'hidden_layer_sizes': layers}
    clf = MLPClassifier()
    return 'MLPC', clf, parameters


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


    preprocessing_functions = [
        get_input_output,
        scaler_preprocess
    ]
    X, y = preprocess(df, preprocessing_functions)

    classifiers_to_explore = [
        get_MLPClassifier_for_grid_search()
    ]
    for name, clf, parameters in classifiers_to_explore:
        gs = GridSearchCV(clf, parameters, n_jobs=2, scoring={"AUC":"roc_auc"}, refit="AUC", verbose=3)

        gs.fit(X, y)
        results = gs.cv_results_
        print(name)
        print_grid_search_results(results)