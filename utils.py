import numpy as np
import torch

epsilon = 1e-8


def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def AP_partial(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    cnt_class_with_no_neg = 0
    cnt_class_with_no_pos = 0
    cnt_class_with_no_labels = 0

    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]

        # Filter out samples without label
        idx = (targets != -1)
        scores = scores[idx]
        targets = targets[idx]
        if len(targets) == 0:
            cnt_class_with_no_labels += 1
            ap[k] = -1
            continue
        elif sum(targets) == 0:
            cnt_class_with_no_pos += 1
            ap[k] = -1
            continue
        if sum(targets == 0) == 0:
            cnt_class_with_no_neg += 1
            ap[k] = -1
            continue
        # compute average precision
        ap[k] = average_precision(scores, targets)

    print('#####DEBUG num -1 classes {} '.format(sum(ap == -1)))
    idx_valid_classes = np.where(ap != -1)[0]
    ap_valid = ap[idx_valid_classes]
    map = 100 * np.mean(ap_valid)

    # Compute macro-map
    targs_macro_valid = targs[:, idx_valid_classes].copy()
    targs_macro_valid[targs_macro_valid <= 0] = 0  # set partial labels as negative
    n_per_class = targs_macro_valid.sum(0)  # get number of targets for each class
    n_total = np.sum(n_per_class)
    map_macro = 100 * np.sum(ap_valid * n_per_class / n_total)

    return ap, map, map_macro, cnt_class_with_no_labels, cnt_class_with_no_neg, cnt_class_with_no_pos


def rankmin(x):
  rank = torch.arange(x.shape[1]).type(x.dtype).to(x.device)
  ranks = torch.zeros_like(x).to(x.device)
  for i in range(x.shape[0]):
    tmp = x[i].argsort()
    ranks[i, tmp] = rank
  return ranks


def spearman_correlation(x, y):
    x_rank = rankmin(x)
    y_rank = rankmin(y)
    
    n = x.size(1)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n ** 2 - 1.0)
    return torch.mean(1.0 - (upper / down)).item()


# def cov(m):
#     m = m.type(torch.double)  # uncomment this line if desired
#     fact = 1.0 / (m.shape[-1] - 1)  # 1 / N
#     m -= torch.mean(m, dim=(1, 2), keepdim=True)
#     mt = torch.transpose(m, 1, 2)  # if complex: mt = m.t().conj()
#     return fact * m.matmul(mt).squeeze()


# def corrcoef(x, y):
#     batch_size = x.shape[0]
#     x = torch.stack((x, y), 1)
#     # calculate covariance matrix of rows
#     c = cov(x)
#     # normalize covariance matrix
#     d = torch.diagonal(c, dim1=1, dim2=2)
#     stddev = torch.pow(d, 0.5)
#     stddev = stddev.repeat(1, 2).view(batch_size, 2, 2)
#     c = c.div(stddev)
#     c = c.div(torch.transpose(stddev, 1, 2))
#     return c[:, 1, 0]


# def compute_rank_correlation(x, y):
#     x, y = rankmin(x), rankmin(y)
#     cor_batch = corrcoef(x, y)
#     return torch.mean(cor_batch).item()


def showCM(cms):
    for i, cm in enumerate(cms):
        print(f"Confusion Matrix for Class {i + 1}")
        print("True \\ Pred", "  0  ", "  1  ")
        print("     0      ", f"{cm[0, 0]:<5}", f"{cm[0, 1]:<5}")
        print("     1      ", f"{cm[1, 0]:<5}", f"{cm[1, 1]:<5}")
        print("\n" + "-" * 20 + "\n")