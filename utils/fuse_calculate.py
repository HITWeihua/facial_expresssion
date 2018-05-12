import numpy as np

if __name__ == '__main__':
    # a = np.arange(12)
    a = np.random.random(12)
    a = a.reshape((3, 4))
    print(a)
    b = np.argsort(a, axis=-1)
    print(b)
    for i in range(len(b)):
        for j in range(len(b[i])):
            a[i][b[i][j]] += j+1
    print(a)