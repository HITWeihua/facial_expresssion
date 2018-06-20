import numpy as np
import os

# base_path = "F:\\files\\facial_expresssion\\summaries\\summaries_graph_1218(2)"
# logits = np.array([])
# labels = np.array([])

def count_num(i, logit_value):
    if labels[i] == 0:
        jieguo[logit_value][0] += 1
    elif labels[i] == 1:
        jieguo[logit_value][1] += 1
    elif labels[i] == 2:
        jieguo[logit_value][2] += 1
    elif labels[i] == 3:
        jieguo[logit_value][3] += 1
    elif labels[i] == 4:
        jieguo[logit_value][4] += 1
    elif labels[i] == 5:
        jieguo[logit_value][5] += 1


def add_number(a):
    b = np.argsort(a, axis=-1)
    # print(b)
    for i in range(len(b)):
        for j in range(len(b[i])):
            # a[i][b[i][j]] += j + 1
            a[i][b[i][j]] += 0.1*j
    # print(a)
    return a


def fuse_logits(dtan, dtgn):
    dtan = add_number(dtan)
    dtgn = add_number(dtgn)
    return 0.5*dtgn+0.5*dtan


if __name__ == '__main__':
    # a = np.arange(12)
    # accuracy_array = []
    logits = np.array([])
    labels = np.array([])
    jieguo = np.zeros((6, 6))
    for fold_num in range(10):
        oulu_label = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/labels_output/dtan_resnet_SE/{}/test_l.txt'.format(fold_num))
        oulu_dtan_logit = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/logits_output/dtan_resnet_SE/{}/logit.txt'.format(fold_num))
        oulu_dtgn_logit = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/logits_output/dtgn_conv_diff/{}/logit.txt'.format(fold_num))
        # oulu_dtgn_logit = np.loadtxt('/home/duheran/ck_logits/{}/logit.txt'.format(fold_num))
        logit = fuse_logits(oulu_dtan_logit, oulu_dtgn_logit)
        label = oulu_label
        logit = np.argmax(logit, axis=1)
        label = np.argmax(label, axis=1)
        logits = np.hstack((logits, logit))
        labels = np.hstack((labels, label))
        # accuracy_array.append(evaluation(logit, oulu_label))
    # print(accuracy_array)
    # print(np.mean(accuracy_array))

    # for fold_num in range(10):
        # base_read_path = os.path.join(base_path, str(fold_num))
        # logit_path = os.path.join(base_read_path, "logit.txt")
        # label_path = os.path.join(base_read_path, "test_l.txt")
        # logit = np.loadtxt(logit_path)
        # label = np.loadtxt(label_path)
        # logit = np.argmax(logit, axis=1)
        # label = np.argmax(label, axis=1)
        # logits = np.hstack((logits, logit))
        # labels = np.hstack((labels, label))

    for i in range(len(logits)):
        if logits[i] == 0:
            count_num(i, 0)
        elif logits[i] == 1:
            count_num(i, 1)
        elif logits[i] == 2:
            count_num(i, 2)
        elif logits[i] == 3:
            count_num(i, 3)
        elif logits[i] == 4:
            count_num(i, 4)
        elif logits[i] == 5:
            count_num(i, 5)

    print(jieguo)
    print(np.sum(jieguo))
    jieguo[0][0] = 63
    jieguo[4][0] = 6
    jieguo[4][4] = 68
    jieguo[0][4] = 7

    print(jieguo)
    print(jieguo / 80)
    print((jieguo[0][0] + jieguo[1][1] + jieguo[2][2] + jieguo[3][3] + jieguo[4][4] + jieguo[5][5]) / 480)