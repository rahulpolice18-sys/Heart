from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError("heart.csv was not found. Place your 1025-row heart.csv dataset in this folder.")

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().dropna()

if "target" not in df.columns:
    raise ValueError("The dataset must contain a target column named 'target'.")

feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
missing = [column for column in feature_names + ["target"] if column not in df.columns]
if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

X = df[feature_names]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit preprocessing on training data only, then save each component.
selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

model = RandomForestClassifier(
    n_estimators=800,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    class_weight=None
)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

joblib.dump(model, BASE_DIR / "heart_model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
joblib.dump(selector, BASE_DIR / "selector.pkl")
print("Saved heart_model.pkl, scaler.pkl, selector.pkl")
