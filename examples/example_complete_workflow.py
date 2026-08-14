# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming
from algorithms.grey_model import GM11
from algorithms.topsis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from algorithms.pca import PCA
from algorithms.curve_fitting import CurveFitting
from utils.data_preprocessor import DataPreprocessor


def run_complete_workflow():
    print("=" * 70)
    print("Mathematical Modeling Competition - Complete Workflow")
    print("=" * 70)
    
    print("\n[1] Data Preprocessing")
    print("-" * 50)
    raw_data = np.array([
        [95, 88, 92, 85, 78],
        [82, 75, 80, 90, 85],
        [78, 82, 75, 70, 92],
        [88, 90, 88, 82, 75],
        [75, 70, 78, 88, 88],
        [92, 85, 90, 78, 80]
    ])
    preprocessor = DataPreprocessor()
    result = preprocessor.process(raw_data, fill_missing=True, normalize=True)
    data = result['processed_data']
    print(f"Original: {raw_data.shape} -> Cleaned: {data.shape}")
    
    print("\n[2] Entropy Weight Method")
    print("-" * 50)
    entropy = EntropyWeight()
    ew_result = entropy.evaluate(data)
    weights = np.array(ew_result['weights'])
    print(f"Weights: {[round(w, 4) for w in weights]}")
    
    print("\n[3] TOPSIS Evaluation")
    print("-" * 50)
    topsis = TOPSIS(weights=weights)
    tp_result = topsis.evaluate(data, types=['benefit'] * 5)
    for scheme_idx, score, rank in tp_result['score_ranking']:
        print(f"Scheme {scheme_idx}: score={score:.6f}, rank={rank}")
    
    print("\n[4] PCA Analysis")
    print("-" * 50)
    pca = PCA(n_components=2)
    pca_result = pca.transform_with_details(data)
    print(f"Dimensions: {data.shape} -> {pca_result['transformed'].shape}")
    print(f"Explained variance: {[round(v, 4) for v in pca_result['explained_variance_ratio']]}")
    
    print("\n[5] GM(1,1) Grey Prediction")
    print("-" * 50)
    scores = np.array(tp_result['scores'])
    gm = GM11()
    gm_result = gm.fit_predict(scores, steps=3)
    print(f"Scores: {[round(s, 4) for s in scores]}")
    print(f"Predictions: {[round(v, 4) for v in gm_result['predicted_values']]}")
    
    print("\n[6] Curve Fitting")
    print("-" * 50)
    x = np.arange(len(scores))
    cf_result = CurveFitting.linear_fit(x, scores)
    print(f"Equation: {cf_result['equation']}")
    print(f"R-squared: {cf_result['r_squared']:.6f}")
    
    print("\n[7] Resource Allocation Optimization")
    print("-" * 50)
    c = -weights
    A_ub = np.array([[1, 1, 1, 1, 1]])
    b_ub = np.array([100])
    bounds = [(0, 30)] * 5
    lp = LinearProgramming()
    opt_result = lp.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print(f"Allocation: {[round(x, 2) for x in opt_result['optimal_solution']]}")
    print(f"Max benefit: {-opt_result['optimal_value']:.6f}")
    
    print("\n" + "=" * 70)
    print("Workflow completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_workflow()