from sklearn.datasets import make_blobs

def load_large(n_samples, n_features):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features)
    # X, y = make_classification(n_samples=n_samples, n_features=n_features, 
    #                            n_informative=n_features, n_redundant=0, n_classes=3, class_sep=10) 
    return X, y
