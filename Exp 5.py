import numpy as np 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X,y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_labeled, y_labeled)

n_iterations = 5
confidence_threshold = 0.7

for i in range(n_iterations):
    if len(X_unlabeled) == 0:
        break
    probs = clf.predict_proba(X_unlabeled)
    preds = clf.predict(X_unlabeled)
    confidence = np.max(probs, axis=1)
    
    confident_indices = np.where(confidence >= confidence_threshold)[0]
    
    if len(confident_indices) == 0:
        print(f"No confident predictions in iteration {i+1}. Stopping early.")
        break
    
    X_new = X_unlabeled[confident_indices]
    y_new = preds[confident_indices]
    X_labeled = np.vstack((X_labeled, X_new))
    y_labeled = np.hstack((y_labeled, y_new))
    
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[confident_indices] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]
    
    clf.fit(X_labeled, y_labeled)
    
    print(f"Iteration {i+1}: labeled data size = {len(X_labeled)}")
    
    if len(X_unlabeled) > 0:
        y_pred = clf.predict(X_unlabeled)
        print("final accuracy on remaining unlabeled data:", accuracy_score(y_unlabeled, y_pred))
    else:
        print("No data left to evaluate.")