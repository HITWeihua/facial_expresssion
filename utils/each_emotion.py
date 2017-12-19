import numpy as np
import os

base_path = "F:\\files\\facial_expresssion\\summaries\\summaries_graph_1218(2)"
logits = np.array([])
labels = np.array([])
jieguo = np.zeros((6, 6))
for fold_num in range(10):
    base_read_path = os.path.join(base_path, str(fold_num))
    logit_path = os.path.join(base_read_path, "logit.txt")
    label_path = os.path.join(base_read_path, "test_l.txt")
    logit = np.loadtxt(logit_path)
    label = np.loadtxt(label_path)
    logit = np.argmax(logit, axis=1)
    label = np.argmax(label, axis=1)
    logits = np.hstack((logits, logit))
    labels = np.hstack((labels, label))


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
print(jieguo/80)
print((jieguo[0][0]+jieguo[1][1]+jieguo[2][2]+jieguo[3][3]+jieguo[4][4]+jieguo[5][5])/480)