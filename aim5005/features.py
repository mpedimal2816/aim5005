import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        return (x - self.minimum) / diff_max_min

    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and standard deviation of each feature.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
           raise RuntimeError("This StandardScaler instance is not fitted yet.")
        # Avoid division by zero by replacing zero std with 1 (or handle as needed)
        std_adj = np.where(self.std == 0, 1, self.std)
        return (X - self.mean)/std_adj 


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)
 
    
class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
        return np.array([np.argwhere(self.classes_ == label).flatten()[0] for label in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)