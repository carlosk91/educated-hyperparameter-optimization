import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, check_dimension
from skopt.utils import dimensions_aslist
from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def fetch_and_prepare_data(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets['y']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded


def split_data(X, y, test_size=0.2, random_state=RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessing_pipeline(categorical_cols, numeric_cols):
    numeric_imputation_mapper = DataFrameMapper(
        [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in numeric_cols],
        input_df=True, df_out=True
    )

    categorical_imputation_mapper = DataFrameMapper(
        [([category_feature], SimpleImputer(strategy="most_frequent")) for category_feature in categorical_cols],
        input_df=True, df_out=True
    )

    return FeatureUnion([
        ("num_mapper", numeric_imputation_mapper),
        ("cat_mapper", categorical_imputation_mapper)
    ])


class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).to_dict("records")


def create_pipeline(X_train):
    preprocessor = build_preprocessing_pipeline(X_train.select_dtypes(include=['object']).columns.tolist(),
                                                X_train.select_dtypes(exclude=['object']).columns.tolist())
    return Pipeline([
        ("featureunion", preprocessor),
        ("dictifier", Dictifier()),
        ("vectorizer", DictVectorizer(sort=False)),
        ("clf", XGBClassifier(random_state=RANDOM_SEED))
    ])


def train_model(pipeline, x, y, search_method, params=None, n_iter=None):
    if search_method == 'baseline':
        searcher = pipeline
    elif search_method == 'grid':
        searcher = GridSearchCV(estimator=pipeline, param_grid=params, scoring='roc_auc', cv=3)
    elif search_method == 'bayes':
        searcher = BayesSearchCV(estimator=pipeline, search_spaces=params, n_iter=n_iter, scoring='roc_auc', cv=3)
    else:
        raise ValueError(f"Invalid search method '{search_method}'. Expected one of: 'baseline', 'grid', 'bayes'.")
    start_time = time.time()
    searcher.fit(x, y)
    end_time = time.time()
    return {"model": searcher, "time": end_time - start_time}


def fit_hyperparameters_model(search_results):
    features = pd.DataFrame(search_results.cv_results_['params'])
    target = search_results.cv_results_['mean_test_score']
    kernel = Matern(nu=1.5)
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=RANDOM_SEED)
    gp_model.fit(features, target)
    return gp_model


def predict_optimal_hyperparameters(gp_model, param_space):
    space = dimensions_aslist(param_space)
    best_hyperparams = [gp_model.X_train_[gp_model.y_train_.argmax(), i] for i in range(gp_model.X_train_.shape[1])]
    param_names = [dimension.name for dimension in space]
    safe_hyperparams = np.clip(best_hyperparams, 0, 1)
    transformed_params = [check_dimension(dimension).inverse_transform([x])[0] for dimension, x in
                          zip(space, safe_hyperparams)]
    optimal_prediction, std_dev = gp_model.predict(np.array(transformed_params).reshape(1, -1), return_std=True)
    optimal_params_dict = dict(zip(param_names, transformed_params))
    return optimal_params_dict, optimal_prediction, std_dev


def calculate_probability_of_improvement(bayes_result, predicted_auc, predicted_std_dev):
    return norm.cdf((predicted_auc - bayes_result['model'].best_score_) / predicted_std_dev)


def plot_roc_curves(model_results, x, y):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("colorblind", len(model_results))
    line_styles = {'Baseline Model': '-',
                   'Grid Search CV Model': '--',
                   'Bayesian Search CV Model': ':',
                   '1 more Gaussian Process on Bayesian Optimized Model': '-.'}

    for result, color in zip(model_results, colors):
        y_pred_proba = result['model'].predict_proba(x)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        line_style = line_styles.get(result['label'], '-')
        plt.plot(fpr, tpr, color=color, lw=1.2, linestyle=line_style, alpha=1,
                 label=f'{result["label"]} (AUC = {roc_auc:.4f}, Time = {result["time"]:,.2f}s)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend(loc="lower right")
    plt.savefig('images/ROC_comparison.png', bbox_inches='tight')
    plt.close()


def print_model_scores(model_results, x, y):
    for model_result in model_results:
        y_pred_prob = model_result["model"].predict_proba(x)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        print(f'{model_result["label"]} AUC: {roc_auc:.4f}. Training time: {model_result["time"]:,.2f}s')


def plot_hyperparameter_heatmap(grid_results, bayes_results):
    grid_params = pd.DataFrame(grid_results.cv_results_['params'])
    grid_scores = pd.Series(grid_results.cv_results_['mean_test_score'])
    bayes_params = pd.DataFrame(bayes_results.cv_results_['params'])
    bayes_scores = pd.Series(bayes_results.cv_results_['mean_test_score'])

    # Get the two most relevant hyperparameters from the Grid Search results
    relevant_params = grid_params.corrwith(grid_scores).abs().sort_values(ascending=False).head(2).index.tolist()

    if len(relevant_params) < 2:
        print("Not enough hyperparameters for heatmap.")
        return

    # Combine the results into a single DataFrame
    grid_params['score'] = grid_scores
    bayes_params['score'] = bayes_scores
    grid_params['method'] = 'Grid Search'
    bayes_params['method'] = 'Bayesian Search'
    combined_params = pd.concat([grid_params, bayes_params], ignore_index=True)

    x_param = relevant_params[0]
    y_param = relevant_params[1]

    # Plot the contour plot
    plt.figure(figsize=(12, 8))

    # Scatter plot of the hyperparameter combinations
    sns.scatterplot(data=combined_params, x=x_param, y=y_param, hue='method', size='score', palette='viridis',
                    legend=False, sizes=(20, 200))

    # Contour plot of the scores
    plt.tricontourf(combined_params[x_param], combined_params[y_param], combined_params['score'], levels=20,
                    cmap="viridis", alpha=0.7)
    plt.colorbar(label='AUC Score')

    plt.title(f"Contour Plot of {x_param} vs {y_param} with AUC Scores (Combined Search Results)")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.savefig('images/hyperparameter_auc_contour.png', bbox_inches='tight')
    plt.close()


def main():
    X, y_encoded = fetch_and_prepare_data(222)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    pipeline = create_pipeline(X_train)
    model_results = []

    param_grid = {'clf__max_depth': [3, 10],
                  'clf__learning_rate': [0.01, 0.2],
                  'clf__n_estimators': [100, 200, 300],
                  'clf__colsample_bytree': [0.5, 1]
                  }

    param_space = {'clf__max_depth': Integer(3, 10),
                   'clf__learning_rate': Real(0.01, 0.2, "log-uniform"),
                   'clf__n_estimators': Integer(50, 300),
                   'clf__colsample_bytree': Real(0.5, 1, "log-uniform")}

    # Baseline Model
    baseline_result = train_model(pipeline, X_train, y_train, 'baseline')
    model_results.append(
        {"model": baseline_result["model"], "label": "Baseline Model", "time": baseline_result["time"]})

    # Cross-Validation Model
    grid_result = train_model(pipeline, X_train, y_train, 'grid', param_grid)
    model_results.append({"model": grid_result["model"], "label": "Grid Search CV Model", "time": grid_result["time"]})

    # Bayesian Optimization Model
    bayes_result = train_model(pipeline, X_train, y_train, 'bayes', param_space, n_iter=24)
    model_results.append(
        {"model": bayes_result["model"], "label": "Bayesian Search CV Model", "time": bayes_result["time"]})

    # Bayesian Optimization, getting the probability of improving the model hyperparameters on a next iteration.
    start_time = time.time()
    hyperparameter_gp_model = fit_hyperparameters_model(bayes_result["model"])
    optimal_params, optimal_prediction, std_dev = predict_optimal_hyperparameters(hyperparameter_gp_model,
                                                                                  param_space)
    gpbo_pipeline = create_pipeline(X_train)
    gpbo_pipeline.set_params(**{key: val for key, val in optimal_params.items()})
    gpbo_pipeline.fit(X_train, y_train)
    final_time = time.time()
    model_results.append({"model": gpbo_pipeline, "label": "1 more Gaussian Process on Bayesian Optimized Model",
                          "time": (final_time - start_time)})

    plot_roc_curves(model_results, X_test, y_test)

    print_model_scores(model_results, X_test, y_test)
    print(f"Optimal Predicted ROC AUC after next iteration: {optimal_prediction[0]:.4f} ± {1.96 * std_dev[0]:.4f}")
    print(f"The probability of finding a better model than the current best on the next iteration is "
          f"{calculate_probability_of_improvement(bayes_result, optimal_prediction[0], std_dev[0]):.2%}")

    plot_hyperparameter_heatmap(grid_result['model'], bayes_result['model'])


if __name__ == "__main__":
    main()
