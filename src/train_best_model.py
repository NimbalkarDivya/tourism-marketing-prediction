import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import load_data, get_preprocessor

# Load data
df = load_data("data/travel1.csv")

# Remove unnecessary columns
df = df.drop(columns=["CustomerID"], errors="ignore")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Main marketing features
selected_features = [
    "MonthlyIncome",
    "CityTier",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "TypeofContact"
]

X = df[selected_features]
y = df["ProdTaken"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Preprocessing
preprocessor = get_preprocessor(X)

# Decision Tree
model = DecisionTreeClassifier(random_state=42)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipe.fit(X_train, y_train)

# Save model
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Decision Tree Model (Main Features) Saved Successfully ✅")