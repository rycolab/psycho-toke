import pandas as pd
import numpy as np

import statsmodels.formula.api as smf

from scipy import stats
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold 


def all_pairs(list):
    """
    Generate all pairs of elements in a list.
    """
    pairs = []
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            pairs.append([list[i], list[j]])
    return pairs


def compute_rsquared(fitted_regressor, data, predicted_var):
    """
    Compute R-squared for a given regressor and dataset.
    """
    y_pred = fitted_regressor.predict(data)
    y_true = data[predicted_var]
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    rsquared = 1 - rss / tss
    return rsquared


def compute_loglik(fitted_regressor, data, predictors, predicted, per_item=True, interactions=False):
    """
    Compute log-likelihood for a given regressor and test set.
    """
    if not interactions:
        dummy_regressor = smf.ols(
            formula=f'{predicted} ~ {" + ".join(predictors)}', 
            data=data
        )
        loglik = dummy_regressor.loglike(fitted_regressor.params)
    else:
        raise NotImplementedError("Interactions not implemented yet.")
    
    if per_item:
        loglik /= len(data)
    
    return loglik


def get_roi_and_fa_type(str):
    """
    Extract ROI and FA type from a string.
    """
    if 'LeadingAndTrailing' in str:
        roi_type = 'LeadingAndTrailing'
    elif 'Leading' in str:
        roi_type = 'Leading'
    elif 'Trailing' in str:
        roi_type = 'Trailing'
    elif 'NoWhitespace' in str:
        roi_type = 'NoWhitespace'
    else:
        raise ValueError(f"Unknown ROI type in {str}")
    
    fa_type = str.split(roi_type)[-1]
    return roi_type, fa_type


def run_crossvalidation(data, LM, predicted_variables, baseline_predictors, baseline_predictors_spillover, surprisal_predictors, surprisal_predictors_spillover, output_path=None, n_seeds=100, n_folds=10, alpha=0.0, L1_wt=1.0):

    # Data normalization
    scaler = MinMaxScaler()
    data_norm = data.copy()
    all_surprisal_predictors = list(set([item for sublist in surprisal_predictors_spillover for item in sublist]))
    data_norm[baseline_predictors + all_surprisal_predictors] = scaler.fit_transform(data[baseline_predictors + all_surprisal_predictors])
    data_norm = data_norm.dropna()

    # Begin cross-validation
    results = []

    for i, predicted_var in enumerate(predicted_variables):
        print(f"{i+1}/{len(predicted_variables)}: {predicted_var}")

        for spillover in [True]: #[False, True]:
            print(f"Spillover: {spillover}")

            if spillover:
                baseline_preds = baseline_predictors_spillover
                surprisal_preds = surprisal_predictors_spillover
            else:
                baseline_preds = baseline_predictors
                surprisal_preds = surprisal_predictors

            # N_SEEDS cross-validation runs
            for seed in range(n_seeds):
                if seed % 20 == 0:
                    print(f"Seed: {seed}/{n_seeds}")

                kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
                kf.get_n_splits(data_norm) 

                # N_FOLDS cross-validation
                for fold, (train_indices, test_indices) in enumerate(kf.split(data_norm)):

                    # Split data into training and test sets
                    df_tmp_fold = data_norm.iloc[train_indices]
                    df_tmp_fold_test = data_norm.iloc[test_indices]

                    # Fit baseline regressor
                    baseline_regressor = smf.ols(
                        formula=f'{predicted_var} ~ {" + ".join(baseline_preds)}', 
                        data=df_tmp_fold
                    )
                    fitted_baseline_regressor = baseline_regressor.fit_regularized(alpha=alpha, L1_wt=L1_wt)

                    baseline_rsquared_train = compute_rsquared(fitted_baseline_regressor, df_tmp_fold, predicted_var)
                    baseline_loglik_train = compute_loglik(fitted_baseline_regressor, df_tmp_fold, baseline_preds, predicted_var, per_item=True)
                    baseline_rsquared_test = compute_rsquared(fitted_baseline_regressor, df_tmp_fold_test, predicted_var)
                    baseline_loglik_test = compute_loglik(fitted_baseline_regressor, df_tmp_fold_test, baseline_preds, predicted_var, per_item=True)

                    # Target regressors
                    for target_preds in surprisal_preds:
                        
                        roi_type, fa_type = get_roi_and_fa_type(target_preds[0])
    
                        # Fit target regressor
                        target_regressors = smf.ols(
                            formula=f'{predicted_var} ~ {" + ".join(baseline_preds)} + {" + ".join(target_preds)}', 
                            data=df_tmp_fold
                        )
                        fitted_target_regressor = target_regressors.fit_regularized(alpha=alpha, L1_wt=L1_wt)

                        # Compute R-squared on test set
                        target_rsquared_train = compute_rsquared(fitted_target_regressor, df_tmp_fold, predicted_var)
                        target_loglik_train = compute_loglik(fitted_target_regressor, df_tmp_fold, baseline_preds + target_preds, predicted_var, per_item=True)
                        target_rsquared_test = compute_rsquared(fitted_target_regressor, df_tmp_fold_test, predicted_var)
                        target_loglik_test = compute_loglik(fitted_target_regressor, df_tmp_fold_test, baseline_preds + target_preds, predicted_var, per_item=True)

                        results.append({
                            "predicted": predicted_var, 
                            "predictor": target_preds[0],
                            "model": LM, 
                            "roi_type": roi_type,
                            "fa_type": fa_type,
                            "spillover": spillover,
                            "fold": f"{seed}_{fold}",
                            "rsquared_train": target_rsquared_train,
                            "delta_rsquared_train": target_rsquared_train - baseline_rsquared_train,
                            "rsquared_test": target_rsquared_test,
                            "delta_rsquared_test": target_rsquared_test - baseline_rsquared_test,
                            "loglik_train": target_loglik_train,
                            "delta_loglik_train": target_loglik_train - baseline_loglik_train,
                            "loglik_test": target_loglik_test,
                            "delta_loglik_test": target_loglik_test - baseline_loglik_test
                        })

    results_df = pd.DataFrame(results)

    if output_path is not None:
        results_df.to_csv(output_path, index=False)

    return results_df


# Statistic
def difference_of_means(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def permutation_test_against_baseline(crossvalidation_results, predicted_variables, predictors, statistic, n_permutations=1000, random_state=0):
    """
    Perform permutation tests between the R^2 of target regressors and baseline regressors.
    """
    pvals = []

    for predicted_var in predicted_variables:
        for target_predictor in predictors:
            target_predictor = target_predictor[0]
            for spillover in [True]:
                crossvalidation_results_spillover = crossvalidation_results[crossvalidation_results.spillover == spillover]
                # Target regressor's R^2 (on the test set, across CV folds and random seeds)
                rsquared_target = crossvalidation_results_spillover[
                    (crossvalidation_results_spillover.predicted == predicted_var)
                ][crossvalidation_results_spillover.predictor == target_predictor].rsquared_test.to_numpy()

                # Delta R^2 of target regressor vs. baseline (on the test set, across CV folds and random seeds)
                delta_rsquared = crossvalidation_results_spillover[
                    (crossvalidation_results_spillover.predicted == predicted_var)
                ][crossvalidation_results_spillover.predictor == target_predictor].delta_rsquared_test.to_numpy()

                # Baseline regressor's R^2 (on the test set, across CV folds and random seeds)
                baseline_rsquared = rsquared_target - delta_rsquared

                # Permutation test
                # Null hypothesis: the target regressor's R^2 is not greater than the baseline regressor's R^2
                # Alternative hypothesis: the target regressor's R^2 is greater than the baseline regressor's R^2
                result = stats.permutation_test(
                    (rsquared_target, baseline_rsquared), 
                    statistic, vectorized=True, n_resamples=n_permutations, alternative='greater', permutation_type='samples', random_state=random_state
                )
                pvals.append({
                    "predicted": predicted_var,
                    "predictor": target_predictor,
                    "spillover": spillover,
                    "pvalue": result.pvalue
                })

    return pd.DataFrame(pvals)


def permutation_test_between_targets(crossvalidation_results, predicted_variables, predictors, statistic, n_permutations=1000, random_state=0):
    """
    Perform permutation tests between the R^2 of target regressors.
    """
    pvals = []

    for predicted_var in predicted_variables:
        for target_predictor1 in predictors:
            target_predictor1 = target_predictor1[0]
            for target_predictor2 in predictors:
                target_predictor2 = target_predictor2[0]
                
                if target_predictor1 == target_predictor2:
                    continue

                for spillover in [True]:

                    crossvalidation_results_spillover = crossvalidation_results[crossvalidation_results.spillover == spillover]

                    # R^2 of first target regressor (on the test set, across CV folds and random seeds)
                    rsquared_target1 = crossvalidation_results_spillover[
                        (crossvalidation_results_spillover.predicted == predicted_var) 
                    ][crossvalidation_results_spillover.predictor == target_predictor1].rsquared_test.to_numpy()

                    # R^2 of second target regressor (on the test set, across CV folds and random seeds)
                    rsquared_target2 = crossvalidation_results_spillover[
                        (crossvalidation_results_spillover.predicted == predicted_var)
                    ][crossvalidation_results_spillover.predictor == target_predictor2].rsquared_test.to_numpy()

                    # Permutation test
                    # Null hypothesis: |the first target regressor's R^2 is not greater than the second target regressor's R^2
                    # Alternative hypothesis: the first target regressor's R^2 is greater than the second target regressor's R^2
                    result = stats.permutation_test(
                        (rsquared_target1, rsquared_target2), 
                        statistic, vectorized=True, n_resamples=n_permutations, alternative='greater', permutation_type='samples', random_state=random_state
                    )
                    pvals.append({
                        "predicted": predicted_var,
                        "predictor1": target_predictor1,
                        "predictor2": target_predictor2,
                        "spillover": spillover,
                        "pvalue": result.pvalue
                    })

    return pd.DataFrame(pvals)
