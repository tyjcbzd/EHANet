import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.spatial.distance import directed_hausdorff


def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def miou_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def S_measure(true_mask, pred_mask, alpha=0.5):
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)

    true_area = np.sum(true_mask)
    pred_area = np.sum(pred_mask)
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    precision = intersection_area / pred_area if pred_area > 0 else 0
    recall = intersection_area / true_area if true_area > 0 else 0

    s_measure = (1 + alpha) * precision * recall / (alpha * precision + recall) if (alpha * precision + recall) > 0 else 0

    return s_measure

def F_measure(true_mask, pred_mask, beta2=0.3):
    # 排除预测或者gt全为0的情况
    if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0:
        return 1.0

    prec = np.sum((pred_mask == 1) & (true_mask == 1)) / (np.sum(pred_mask == 1) + 1e-20)
    recall = np.sum((pred_mask == 1) & (true_mask == 1)) / (np.sum(true_mask == 1) + 1e-20)

    score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
    return score


def E_phi(true_mask, pred_mask):
    intersection = np.logical_and(true_mask, pred_mask)

    true_positive = np.sum(intersection)
    false_positive = np.sum(pred_mask) - true_positive
    false_negative = np.sum(true_mask) - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    e_phi = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return e_phi


def mean_absolute_error(true_mask, pred_mask):
    abs_error = np.abs(true_mask - pred_mask)
    mean_error = np.mean(abs_error)

    return mean_error



def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = miou_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred)
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        score_specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    else:
        score_specificity = 0.0

    return [score_jaccard, score_f1, score_recall, score_precision, score_specificity, score_acc, score_fbeta]

def cal_hd_dis(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    #
    y_true = y_true.squeeze(0).squeeze(0)
    y_pred = y_pred.squeeze(0).squeeze(0)

    hd = directed_hausdorff(y_true, y_pred)[0]
    return hd
