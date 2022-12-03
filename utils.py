import torch
import numpy as np
import torch.nn.functional as F


def get_num_params(model):
    r"""

    Description
    -----------
    Convert scipy sparse matrix to torch sparse tensor.

    Parameters
    ----------
    model : torch.nn.module
        Model implemented based on ``torch.nn.module``.

    """
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def eval_acc(pred, dataset, verbose=True):
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid
    train_acc = torch.sum(torch.argmax(pred[train_nid], dim=1) == dataset.data.y[train_nid]).item() / len(train_nid)
    val_acc = torch.sum(torch.argmax(pred[val_nid], dim=1) == dataset.data.y[val_nid]).item() / len(val_nid)
    test_acc = torch.sum(torch.argmax(pred[test_nid], dim=1) == dataset.data.y[test_nid]).item() / len(test_nid)

    if verbose:
        print("ACC: Train {:.4f} | Val {:.4f} | Test {:.4f}".format(train_acc, val_acc, test_acc))

    return train_acc, val_acc, test_acc


def eval_confidence(pred, dataset, verbose=True, softmax=True):
    # evaluate confidence
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid
    if softmax:
        train_confidence = torch.max(torch.softmax(pred[train_nid], dim=1), dim=1).values
        val_confidence = torch.max(torch.softmax(pred[val_nid], dim=1), dim=1).values
        test_confidence = torch.max(torch.softmax(pred[test_nid], dim=1), dim=1).values
    else:
        train_confidence = torch.max(pred[train_nid], dim=1).values
        val_confidence = torch.max(pred[val_nid], dim=1).values
        test_confidence = torch.max(pred[test_nid], dim=1).values
    train_mean = torch.mean(train_confidence).item()
    train_std = torch.std(train_confidence).item()
    val_mean = torch.mean(val_confidence).item()
    val_std = torch.std(val_confidence).item()
    test_mean = torch.mean(test_confidence).item()
    test_std = torch.std(test_confidence).item()

    if verbose:
        print(
            "Confidence: Train {:.4f}+-{:.4f} | Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}".format(train_mean, train_std,
                                                                                                 val_mean, val_std,
                                                                                                 test_mean, test_std))

    return train_confidence, val_confidence, test_confidence


def eval_entropy(pred, dataset, verbose=True, softmax=True):
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid
    if softmax:
        train_prob = torch.softmax(pred[train_nid], dim=1)
        val_prob = torch.softmax(pred[val_nid], dim=1)
        test_prob = torch.softmax(pred[test_nid], dim=1)
    else:
        train_prob = pred[train_nid]
        val_prob = pred[val_nid]
        test_prob = pred[test_nid]
    train_entropy = torch.sum(-train_prob * torch.log(train_prob), dim=1)
    val_entropy = torch.sum(-val_prob * torch.log(val_prob), dim=1)
    test_entropy = torch.sum(-test_prob * torch.log(test_prob), dim=1)

    if verbose:
        print("Entropy: Train {:.4f}+-{:.4f} | Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}".format(
            torch.mean(train_entropy).item(), torch.std(train_entropy).item(),
            torch.mean(val_entropy).item(), torch.std(val_entropy).item(),
            torch.mean(test_entropy).item(), torch.std(test_entropy).item()
            ))

    return train_entropy, val_entropy, test_entropy


def eval_perplexity(pred, dataset, verbose=True):
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid
    labels = dataset.data.y
    train_perp = torch.exp(F.cross_entropy(pred[train_nid], labels[train_nid])).item()
    val_perp = torch.exp(F.cross_entropy(pred[val_nid], labels[val_nid])).item()
    test_perp = torch.exp(F.cross_entropy(pred[test_nid], labels[test_nid])).item()

    if verbose:
        print("Perplexity: Train {:.4f} | Val {:.4f} | Test {:.4f}".format(
            train_perp, val_perp, test_perp))

    return train_perp, val_perp, test_perp
