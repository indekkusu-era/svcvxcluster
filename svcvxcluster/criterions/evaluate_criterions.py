def evaluate_criterions(X, B, Z, A, eps, C, mu, list_criterions, **kwargs):
    return list(map(lambda crit: crit(X=X, B=B, Z=Z, A=A, eps=eps, C=C, mu=mu, **kwargs), list_criterions))
