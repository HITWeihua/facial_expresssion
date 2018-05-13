import numpy as np


def add_number(a):
    b = np.argsort(a, axis=-1)
    # print(b)
    for i in range(len(b)):
        for j in range(len(b[i])):
            a[i][b[i][j]] += j + 1
    # print(a)
    return a


def fuse_logits(dtan, dtgn):
    dtan = add_number(dtan)
    dtgn = add_number(dtgn)
    return 0.5*dtgn+0.5*dtan


def evaluation(logits, labels):
    correct_prediction = np.equal(np.argmax(logits, 1), np.argmax(labels, 1))
    accuracy = np.mean(correct_prediction.astype('float'))
    return accuracy


if __name__ == '__main__':
    # a = np.arange(12)
    accuracy_array = []
    for fold_num in range(10):
        oulu_label = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/labels_output/dtan_resnet_SE/{}/test_l.txt'.format(fold_num))
        oulu_dtan_logit = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/logits_output/dtan_resnet_SE/{}/logit.txt'.format(fold_num))
        # oulu_dtgn_logit = np.loadtxt('/home/duheran/facial_expresssion/prcv_expreiment/trainging/logits_output/dtgn_conv_diff/{}/logit.txt'.format(fold_num))
        oulu_dtgn_logit = np.loadtxt('/home/duheran/oulu_logits/{}/logit.txt'.format(fold_num))
        logit = fuse_logits(oulu_dtan_logit, oulu_dtgn_logit)

        accuracy_array.append(evaluation(logit, oulu_label))
    print(accuracy_array)
    print(np.mean(accuracy_array))





