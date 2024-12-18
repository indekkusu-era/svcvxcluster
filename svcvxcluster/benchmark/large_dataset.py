from sklearn.datasets import make_blobs

def load_large(n_samples, n_features):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features)
    return X, y
