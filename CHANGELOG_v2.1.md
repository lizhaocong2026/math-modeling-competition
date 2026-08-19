# Changelog

## v2.1.0 - 2026-08-19 Deep Brainstorm Enhancement

### New Algorithms (10)
- lgorithms/svr.py - SVM Regression
- lgorithms/xgboost.py - XGBoost Regressor
- lgorithms/lightgbm.py - LightGBM Regressor
- lgorithms/lda.py - Linear Discriminant Analysis
- lgorithms/stl_decompose.py - STL Time Series Decomposition
- lgorithms/sarima.py - SARIMA Seasonal Model
- lgorithms/rf_regression.py - Random Forest Regressor
- lgorithms/cma_es.py - CMA-ES Evolution Strategy
- lgorithms/gan.py - Generative Adversarial Network
- lgorithms/automl.py - AutoML Pipeline

### New Paper Templates
- paper/document.tex - Main compilation file
- paper/texfile/1abstract.tex through 9Appendix.tex - 9-section structure
- Complete LaTeX paper template with AI usage declaration module

### New Documentation
- docs/brainstorming_report.md - Deep brainstorming analysis report
- docs/paper_writing/paper_writing_guide.md - Paper writing guide
- docs/references/external_resources.md - External resources summary
- 
eference_papers/README.md - Reference papers repository guide

### External Resources Collected
- 5 LaTeX template repositories
- 8 comprehensive math modeling repos
- 4 past competition paper repositories
- Total: 17 external resources documented

### Stats
- Algorithms: 56 -> 66
- Cases: 6 (unchanged, will add 4 more in next phase)
- Templates: 6 Python + 10 LaTeX
- Documentation: 20 -> 25 files
- Tests: 53 (will add 15 more for new algorithms)

### Known Issues
- XGBoost/LightGBM require additional dependencies (pip install xgboost lightgbm)
- STL/SARIMA require statsmodels (pip install statsmodels)
- GAN requires PyTorch (pip install torch)
