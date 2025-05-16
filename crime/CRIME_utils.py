
# import tensorflow as tf
import numpy as np

# Helper functions within functions




def cosine_similarity_manual(v1, v2):

    """
    Calculate cosine similarity between spectra.

    Parameters:
    - v1: array of spectra 1
    - v1: array of spectra 2

    Returns:
    - scos: cosine similarity
    """

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    scos = dot_product / (norm_v1 * norm_v2)
    return scos


def discretize_concentration(concentrations, thresholds=(3, 6)):
    """
    Discretize continuous concentration levels into categorical bins.

    Parameters:
    - concentrations: array of concentration values
    - thresholds: tuple containing the thresholds for categorization

    Returns:
    - discretized_labels: array of discretized concentration levels
    """

    discretized_labels = []
    for concentration in concentrations:
        if concentration == 0:
            label = 0  # No concentration
        elif concentration <= thresholds[0]:
            label = 1  # Low concentration
        elif concentration <= thresholds[1]:
            label = 2  # Medium concentration
        else:
            label = 3  # High concentration
        discretized_labels.append(label)
    
    return discretized_labels



from sklearn.decomposition import PCA

class PCAEncoder:
    def __init__(self, n_components=2, random_state=None):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.fitted = False

    def fit(self, X):
        """
        X: np.ndarray of shape (n_samples, n_perturbs, n_features)
        """
        # flatten each sample
        flat = X.reshape(X.shape[0], -1)
        self.pca.fit(flat)
        self.fitted = True
        return self

    def predict(self, X):
        """
        X: same shape as in fit
        returns: (Z, X_recon)
          Z:   (n_samples, n_components)
          X_recon: (n_samples, n_perturbs, n_features)  inverse‐transformed back
        """
        if not self.fitted:
            raise RuntimeError("PCAEncoder must be fit() before predict()")
        flat = X.reshape(X.shape[0], -1)
        Z = self.pca.transform(flat)                      # (n,2)
        flat_recon = self.pca.inverse_transform(Z)        # (n, n_perturbs * n_features)
        X_recon = flat_recon.reshape(X.shape)             # back to (n, n_perturbs, n_features)
        return Z, X_recon

