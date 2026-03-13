"""
Classical ML Models
1. Logistic Regression (custom)
2. SVM
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


def probability_uncertainty(probability: float) -> float:
    """
    Uncertainty score in [0, 1].
    1.0 means highly uncertain (near 0.5), 0.0 means very confident.
    """
    p = float(np.clip(probability, 0.0, 1.0))
    return 1.0 - abs(2.0 * p - 1.0)


class LogisticRegressionScratch:
    """Logistic Regression (custom implementation)"""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, random_state: int = 42, class_weight: float = 1.0):
        """
        Initialize Logistic Regression
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            random_state: Random seed
            class_weight: Weight for positive class (depressed) to handle imbalanced data
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.is_trained = False

    def _to_numpy(self, X):
        """Convert sparse/dense matrices to ndarray for stable math ops."""
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _initialize_weights(self, n_features):
        """Initialize weights and bias"""
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def train(self, X_train, y_train, verbose: bool = True):
        """
        Train using gradient descent
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples]
            verbose: Print training progress
        """
        X_train = self._to_numpy(X_train)
        y_train = np.asarray(y_train).astype(float)
        
        # Validate inputs
        if len(X_train) == 0:
            raise ValueError("Empty training data")
        if len(y_train) == 0:
            raise ValueError("Empty labels")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train ({len(X_train)}) and y_train ({len(y_train)}) must have same length")
        
        n_samples, n_features = X_train.shape
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Training loop
        prev_loss = float('inf')
        for iteration in range(self.max_iter):
            # Forward pass
            z = np.dot(X_train, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute loss (binary cross entropy) with class weights
            # Apply class weight to positive class (depressed)
            weights_array = np.where(y_train == 1, self.class_weight, 1.0)
            loss = -np.mean(weights_array * (y_train * np.log(predictions + 1e-15) + 
                          (1 - y_train) * np.log(1 - predictions + 1e-15)))
            
            # Backward pass (gradient descent) with class weights
            weighted_error = weights_array * (predictions - y_train)
            dw = (1 / n_samples) * np.dot(X_train.T, weighted_error)
            db = (1 / n_samples) * np.sum(weighted_error)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.4f}")
            
            # Early stopping if loss doesn't improve
            if abs(prev_loss - loss) < 1e-6:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_loss = loss
        
        self.is_trained = True
        if verbose:
            print("Logistic Regression trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self._to_numpy(X)
        z = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(z)
        return (predictions >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self._to_numpy(X)
        z = np.dot(X, self.weights) + self.bias
        prob_class_1 = self._sigmoid(z).flatten()
        prob_class_0 = 1.0 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])

    def predict_uncertainty(self, X):
        """Return per-sample uncertainty scores based on class-1 probability."""
        probs = self.predict_proba(X)[:, 1]
        return np.array([probability_uncertainty(p) for p in probs])
    
    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save(self, filepath: str):
        """Save model to file"""
        filepath = os.path.normpath(os.path.abspath(filepath))
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'learning_rate': self.learning_rate,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
                'class_weight': self.class_weight
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file"""
        filepath = os.path.normpath(os.path.abspath(filepath))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.bias = data['bias']
        self.learning_rate = data['learning_rate']
        self.max_iter = data['max_iter']
        self.random_state = data['random_state']
        self.class_weight = data.get('class_weight', 1.0)  # Backward compatibility
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class SVMScratch:
    """Support Vector Machine (custom implementation)"""
    
    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01, 
                 max_iter: int = 1000, random_state: int = 42, class_weight: float = 1.0):
        """
        Initialize SVM
        
        Args:
            learning_rate: Learning rate for gradient descent
            lambda_param: Regularization parameter
            max_iter: Maximum number of iterations
            random_state: Random seed
            class_weight: Weight for positive class (depressed) to handle imbalanced data
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.is_trained = False
    
    def _hinge_loss(self, y, y_pred):
        """Hinge loss function"""
        return np.maximum(0, 1 - y * y_pred)
    
    def _initialize_weights(self, n_features):
        """Initialize weights and bias"""
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def train(self, X_train, y_train, verbose: bool = True):
        """
        Train using gradient descent with hinge loss
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels (converted to -1, 1)
            verbose: Print training progress
        """
        # Convert to numpy array
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Validate inputs
        if len(X_train) == 0:
            raise ValueError("Empty training data")
        if len(y_train) == 0:
            raise ValueError("Empty labels")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train ({len(X_train)}) and y_train ({len(y_train)}) must have same length")
        
        # Convert labels to -1 and 1
        y_train_svm = np.where(y_train == 0, -1, 1)
        
        n_samples, n_features = X_train.shape
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Training loop
        prev_loss = float('inf')
        for iteration in range(self.max_iter):
            # Forward pass
            y_pred = np.dot(X_train, self.weights) + self.bias
            
            # Compute hinge loss
            loss = np.mean(self._hinge_loss(y_train_svm, y_pred)) + \
                   self.lambda_param * np.dot(self.weights, self.weights)
            
            # Compute gradients with class weights
            dw = np.zeros(n_features)
            db = 0
            
            for i in range(n_samples):
                if y_train_svm[i] * y_pred[i] < 1:
                    # Apply class weight: positive class (y=1) gets class_weight, negative (y=-1) gets 1.0
                    weight = self.class_weight if y_train_svm[i] == 1 else 1.0
                    dw += -weight * y_train_svm[i] * X_train[i]
                    db += -weight * y_train_svm[i]
            
            dw = dw / n_samples
            db = db / n_samples
            
            # Add regularization
            dw += 2 * self.lambda_param * self.weights
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.4f}")
            
            # Early stopping if loss doesn't improve
            if abs(prev_loss - loss) < 1e-6:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_loss = loss
        
        self.is_trained = True
        if verbose:
            print("SVM trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        y_pred = np.dot(X, self.weights) + self.bias
        # Convert -1/1 back to 0/1
        return (y_pred >= 0).astype(int)
    
    def decision_function(self, X):
        """Get decision function values (distance from hyperplane)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return np.dot(X, self.weights) + self.bias
    
    def predict_proba(self, X):
        """
        Predict probabilities by converting decision function to probabilities.
        Uses sigmoid function to convert decision values to probabilities.
        This approximates Platt scaling for probability calibration.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get decision function values (distance from hyperplane)
        decision_values = self.decision_function(X)
        
        # Clip values to prevent overflow in sigmoid
        # Most sigmoid activity happens in [-10, 10] range
        decision_values = np.clip(decision_values, -10, 10)
        
        # Apply sigmoid: P(y=1) = 1 / (1 + exp(-decision_value))
        # For SVM trained with -1/1 labels:
        # - Positive decision values indicate class 1 (depressed)
        # - Negative decision values indicate class 0 (non-depressed)
        # The sigmoid converts decision values to probabilities
        # Using stable sigmoid implementation
        exp_neg_dv = np.exp(-decision_values)
        probabilities = 1 / (1 + exp_neg_dv)
        
        # Ensure probabilities are in [0, 1] range (should already be, but double-check)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        # Return as 2D array to match sklearn format: [P(class=0), P(class=1)]
        # For binary classification, we return [1-prob, prob]
        prob_class_1 = probabilities.flatten()
        prob_class_0 = 1 - prob_class_1
        
        # Return shape: (n_samples, 2)
        return np.column_stack([prob_class_0, prob_class_1])
    
    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save(self, filepath: str):
        """Save model to file"""
        filepath = os.path.normpath(os.path.abspath(filepath))
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'learning_rate': self.learning_rate,
                'lambda_param': self.lambda_param,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
                'class_weight': self.class_weight
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file"""
        filepath = os.path.normpath(os.path.abspath(filepath))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.bias = data['bias']
        self.learning_rate = data['learning_rate']
        self.lambda_param = data['lambda_param']
        self.max_iter = data['max_iter']
        self.random_state = data['random_state']
        self.class_weight = data.get('class_weight', 1.0)  # Backward compatibility
        self.is_trained = True
        print(f"Model loaded from {filepath}")
