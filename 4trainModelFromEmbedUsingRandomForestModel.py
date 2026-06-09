import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = np.load("embeddings.npy")
y = np.load("labels.npy")
print("Embeddings :",  X.shape)
print("Labels : " , y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test,predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test,predictions))

joblib.dump(classifier , "emotion_classifier.pkl")
print("\nClassifier saved as emotion_classifier.pkl")