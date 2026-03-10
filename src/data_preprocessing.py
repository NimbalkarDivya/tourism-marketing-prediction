import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

# ------------------------------------------------
# Preprocessor
# ------------------------------------------------
def get_preprocessor(X):

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )

    return preprocessor