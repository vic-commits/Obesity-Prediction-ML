import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def split_data(df: pd.DataFrame, target: "NObeyesdad"):
    df = read_data(df)
    X = df.drop(columns=[target])
    y = LabelEncoder().fit_transform(df[target])
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_features:
        df[col] = LabelEncoder.fit_transform(df[col])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    scaler = None
    X_train_scaled = X_train
    X_test_scaled = X_test
    if scale: 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
