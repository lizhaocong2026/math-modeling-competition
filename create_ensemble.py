import os
base = r"algorithms"

# 1. ensemble_methods.py
with open(os.path.join(base, "ensemble_methods.py"), "w", encoding="utf-8") as f:
    f.write("""# Ensemble Methods Algorithm
import numpy as np
from typing import Dict, List, Any

class EnsembleMethods:
    \"\"\"集成学习方法集合\"\"\"
    
    def __init__(self):
        self.results = {}
    
    def random_forest_classifier(self, X, y, n_estimators=100, max_depth=None, random_state=None):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X, y)
        return {
            'model': rf,
            'predictions': rf.predict(X),
            'accuracy': rf.score(X, y),
            'feature_importances': rf.feature_importances_
        }
    
    def random_forest_regressor(self, X, y, n_estimators=100, max_depth=None, random_state=None):
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X, y)
        return {
            'model': rf,
            'predictions': rf.predict(X),
            'r2_score': rf.score(X, y),
            'feature_importances': rf.feature_importances_
        }
    
    def gradient_boosting_classifier(self, X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        gb.fit(X, y)
        return {'model': gb, 'predictions': gb.predict(X), 'accuracy': gb.score(X, y)}
    
    def gradient_boosting_regressor(self, X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        gb.fit(X, y)
        return {'model': gb, 'predictions': gb.predict(X), 'r2_score': gb.score(X, y)}
    
    def adaboost_classifier(self, X, y, n_estimators=50, learning_rate=1.0):
        from sklearn.ensemble import AdaBoostClassifier
        ab = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        ab.fit(X, y)
        return {'model': ab, 'predictions': ab.predict(X), 'accuracy': ab.score(X, y)}
    
    def bagging_classifier(self, X, y, n_estimators=10, random_state=None):
        from sklearn.ensemble import BaggingClassifier
        bc = BaggingClassifier(n_estimators=n_estimators, random_state=random_state)
        bc.fit(X, y)
        return {'model': bc, 'predictions': bc.predict(X), 'accuracy': bc.score(X, y)}
    
    def voting_classifier(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        lr = LogisticRegression(max_iter=1000)
        svm = SVC(kernel="rbf")
        knn = KNeighborsClassifier()
        from sklearn.ensemble import VotingClassifier
        vc = VotingClassifier(estimators=[("lr", lr), ("svm", svm), ("knn", knn)], voting="soft")
        vc.fit(X, y)
        return {'model': vc, 'predictions': vc.predict(X), 'accuracy': vc.score(X, y)}
    
    def stacking_classifier(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        lr = LogisticRegression(max_iter=1000)
        svm = SVC(kernel="rbf", probability=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        sc = StackingClassifier(estimators=[("svm", svm), ("rf", rf)], final_estimator=lr)
        sc.fit(X, y)
        return {'model': sc, 'predictions': sc.predict(X), 'accuracy': sc.score(X, y)}
    
    def compare_ensemble_methods(self, X, y):
        results = {}
        methods = [
            ("Random Forest", lambda: self.random_forest_classifier(X, y)),
            ("Gradient Boosting", lambda: self.gradient_boosting_classifier(X, y)),
            ("AdaBoost", lambda: self.adaboost_classifier(X, y)),
            ("Voting", lambda: self.voting_classifier(X, y)),
            ("Stacking", lambda: self.stacking_classifier(X, y)),
        ]
        for name, method in methods:
            try:
                res = method()
                results[name] = {"accuracy": res["accuracy"], "predictions": res["predictions"]}
            except Exception as e:
                results[name] = {"error": str(e)}
        return results
""")
print("Created ensemble_methods.py")
