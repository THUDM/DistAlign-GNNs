import torch


def gram_linear(X):
    return X @ X.T


def gram_rbf(X, threshold=1.0):
    gram = gram_linear(X)
    norms = torch.diag(gram)
    dist = -2 * gram + norms[:, None] + norms[None, :]
    dist_median = torch.median(dist)
    rbf = torch.exp(-dist / (2 * threshold ** 2 * dist_median))

    return rbf


def center_gram(gram):
    means = torch.mean(gram, dim=0)
    means -= torch.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

    return gram


def cka(X, Y, mode="linear", threshold=1.0):
    if mode == "linear":
        gram_X = gram_linear(X)
        gram_Y = gram_linear(Y)
    elif mode == "rbf":
        gram_X = gram_rbf(X, threshold)
        gram_Y = gram_rbf(Y, threshold)
    else:
        raise ValueError("Unknown mode {}".format(mode))

    gram_X = center_gram(gram_X)
    gram_Y = center_gram(gram_Y)
    scaled_hsic = gram_X.ravel() @ gram_Y.ravel()
    norm_X = torch.linalg.norm(gram_X)
    norm_Y = torch.linalg.norm(gram_Y)
    rst = scaled_hsic / (norm_X * norm_Y)

    return rst


def cca(X, Y):
    Qx, _ = torch.linalg.qr(X)
    Qy, _ = torch.linalg.qr(Y)
    rst = torch.linalg.norm(Qx.T @ Qy) ** 2 / min(X.shape[1], Y.shape[1])

    return rst
