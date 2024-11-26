import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load


class MovementClassifier:
    def __init__(self, n_estimators=100, max_samples=0.8, max_features=0.8):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=42,
            bootstrap=True,
            n_jobs=-1
        )
        self.is_trained = False
        self.classes_ = None

    def preprocess_data(self, X):
        expected_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        if isinstance(X, pd.DataFrame):
            X = X[expected_columns].values
        return X

    def train(self, data_path, n_splits=5, verbose=True):
        """Train the model with stratified cross-validation."""
        if verbose:
            print("Loading data...")

        # Load data
        df = pd.read_csv(data_path)
        if verbose:
            print(f"Dataset contains {len(df)} samples")

        # Class distribution
        if verbose:
            print("\nClass distribution:")
            print(df['label'].value_counts())

        # Separate features and labels
        X = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
        y = df['label']

        self.classes_ = np.unique(y)

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_train_scores = []
        cv_test_scores = []

        if verbose:
            print("\nStarting cross-validation...")

        # Cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            if verbose:
                print(f"\nTraining fold {fold}/{n_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_train_processed = self.preprocess_data(X_train)
            X_test_processed = self.preprocess_data(X_test)

            if fold == 1:
                self.scaler.fit(X_train_processed)

            X_train_scaled = self.scaler.transform(X_train_processed)
            X_test_scaled = self.scaler.transform(X_test_processed)

            self.model.fit(X_train_scaled, y_train)

            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            cv_train_scores.append(train_score)
            cv_test_scores.append(test_score)

            if verbose:
                print(f"Fold {fold} Training Accuracy: {train_score:.4f}")
                print(f"Fold {fold} Testing Accuracy: {test_score:.4f}")

            if fold == n_splits:
                y_pred = self.model.predict(X_test_scaled)
                if verbose:
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred))
                    print("\nConfusion Matrix:")
                    print(confusion_matrix(y_test, y_pred))

        if verbose:
            print("\nAverage Training Accuracy: {:.4f} ± {:.4f}".format(
                np.mean(cv_train_scores), np.std(cv_train_scores)))
            print("Average Testing Accuracy: {:.4f} ± {:.4f}".format(
                np.mean(cv_test_scores), np.std(cv_test_scores)))

        # Final training on full dataset
        if verbose:
            print("\nTraining final model on full dataset...")
        X_processed = self.preprocess_data(X)
        X_scaled = self.scaler.transform(X_processed)
        self.model.fit(X_scaled, y)

        self.is_trained = True

    def predict(self, features):
        """Make predictions on new data."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        X = self.preprocess_data(features)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return prediction, probabilities

    def save_model(self, model_path='movement_model.joblib', scaler_path='scaler.joblib', skl_path='movement_model.skl'):
        """Save the trained model and scaler to files."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        # Save in .joblib
        dump(self.model, model_path)
        dump(self.scaler, scaler_path)

        # Save in .skl format
        skl_data = {'model': self.model, 'scaler': self.scaler}
        dump(skl_data, skl_path)
        print(f"Model saved in .joblib and .skl formats: {model_path}, {skl_path}")

    def load_model_from_skl(self, skl_path='movement_model.skl'):
        """Load the trained model and scaler from a .skl file."""
        skl_data = load(skl_path)
        self.model = skl_data['model']
        self.scaler = skl_data['scaler']
        self.is_trained = True
        print(f"Model loaded from {skl_path}")


# Main script
if __name__ == "__main__":
    # Initialize the classifier
    classifier = MovementClassifier(
        n_estimators=100,
        max_samples=0.7,
        max_features=0.7
    )

    # Train the model
    classifier.train(
        data_path=r'C:\Users\GiacomoAmato\PycharmProjects\pythonProject1\venv\Scripts\IOT\dataset\tact_base_move.csv',
        verbose=True
    )

    # Save the model
    classifier.save_model(
        model_path='movement_model.joblib',
        scaler_path='scaler.joblib',
        skl_path='movement_model.skl'
    )

    classifier.load_model_from_skl(skl_path='movement_model.skl')
    new_data = pd.DataFrame({'ax': [0.1], 'ay': [-0.2], 'az': [0.3], 'gx': [0.4], 'gy': [-0.5], 'gz': [0.6]})
    prediction, probabilities = classifier.predict(new_data)
    print(f"Prediction: {prediction[0]}, Probabilities: {probabilities[0]}")