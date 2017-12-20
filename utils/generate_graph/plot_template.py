# -*- coding: utf-8 -*-
# by Heran Du
"""
以下代码用于为python生成线性图像的模板函数 基于python3.5, matplotlib2.0
图像中的颜色 折线和点的类型为随机生成 使用者也可以指定
"""

import matplotlib.pyplot as plt
import random
import os
from sklearn.linear_model import LinearRegression

COLORS = (
'black', 'gray', 'silver', 'firebrick', 'r', 'sienna', 'darksalmon', 'gold', 'olivedrab', 'green', 'mediumseagreen'
 , 'skyblue', 'darkblue', 'slateblue', 'blueviolet', 'purple', 'hotpink', 'pink', 'lightcoral', 'maroon'
 , 'coral', 'darkorange', 'peachpuff', 'orange', 'darkgoldenrod', 'lemonchiffon', 'olive', 'yellowgreen', 'forestgreen',
 'g'
 , 'springgreen', 'cyan', 'darkslategray', 'powderblue', 'mediumblue', 'darkslateblue', 'indigo', 'thistle',
 'darkmagenta'
 , 'orchid', 'lightpink', 'dimgray', 'indianred', 'darkred', 'salmon', 'orangered', 'chocolate', 'peru', 'goldenrod', 'y'
 , 'darkolivegreen', 'limegreen', 'turquoise', 'teal', 'lightblue', 'steelblue', 'cornflowerblue', 'midnightblue'
 , 'blue', 'mediumslateblue', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'brown', 'red', 'tomato',
'lightsalmon', 'saddlebrown', 'yellow', 'greenyellow', 'darkgreen', 'seagreen', 'lightseagreen', 'darkcyan',
'deepskyblue'
'darkturquoise', 'slategrey', 'royalblue', 'navy', 'b', 'mediumpurple', 'darkviolet', 'violet', 'fuchsia', 'deeppink',
'crimson')
COLORS2 = (
 'black', 'silver', 'rosybrown', 'firebrick', 'darksalmon', 'sandybrown', 'tan', 'gold', 'darkkhaki', 'olivedrab',
 'darkgreen', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred',
 'palevioletred', 'k', 'grey', 'lightcoral', 'maroon', 'coral', 'peachpuff', 'darkorange', 'navajowhite',
 'darkgoldenrod', 'olive', 'yellowgreen', 'g', 'slategrey', 'slateblue', 'darkblue', 'rebeccapurple',
 'darkviolet', 'violet', 'deeppink', 'crimson', 'dimgray', 'darkgray', 'indianred', 'salmon', 'orangered'
 , 'burlywood', 'goldenrod', 'khaki', 'y', 'darkolivegreen', 'green', 'forestgreen', 'dodgerblue', 'c', 'darkcyan'
 , 'darkslateblue', 'blueviolet', 'purple', 'mediumblue', 'mediumorchid', 'hotpink', 'pink', 'dimgrey', 'darkgrey',
 'brown', 'tomato', 'lightsalmon', 'limegreen', 'teal', 'lightblue', 'steelblue', 'lightslategrey', 'cornflowerblue',
 'b', 'mediumslateblue', 'indigo', 'thistle', 'darkmagenta', 'orchid', 'lightpink')
UN_FILLED_MARKERS = [',', '.', '1', '2', '3', '4', '*', '+', 'x', 'd', 'D', '|', '_']
FILLED_MARKERS = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
LINE_STYLES = [':', '-.', '--', '-', '']


def draw_line_chart(graph_path, title, xlable, ylable, *args):
    """
    画出折线图的模板函数
    :param graph_path: 图片保存的路径和名称
    :param title: 图表的标题
    :param xlable: 图表横轴的名称
    :param ylable: 图表纵轴的名称
    :param args: 图标中每条折现的信息以及颜色，点的类型等属性
    :return: None

    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(1)
    colors = random.sample(COLORS2, len(args))
    for i in range(len(args)):
        if len(args[i]) == 3:
            plt.plot(args[i][0], args[i][1], color=colors[i], marker='.', markersize='5', ls='--', label=args[i][2])
        elif len(args[i]) == 4:
            plt.plot(args[i][0], args[i][1], color=colors[i], marker=args[i][3], markersize='5', ls='--',
                     label=args[i][2])
        elif len(args[i]) == 5:
            plt.plot(args[i][0], args[i][1], color=colors[i], marker=args[i][3], markersize=args[i][4], ls='--',
                     label=args[i][2])
        elif len(args[i]) == 6:
            plt.plot(args[i][0], args[i][1], color=colors[i], marker=args[i][3], markersize=args[i][4], ls=args[i][5],
                     label=args[i][2])
        elif len(args[i]) == 7:
            plt.plot(args[i][0], args[i][1], color=args[i][6], marker=args[i][3], markersize=args[i][4], ls=args[i][5],
                     label=args[i][2])
    plt.title(title, fontsize=17)
    plt.xlabel(xlable, fontsize=10)
    plt.ylabel(ylable, fontsize=10)
    plt.legend(loc=4, fontsize=10, shadow=True)
    plt.savefig(graph_path, dpi=500, bbox_inches='tight')
    plt.close(1)


def create_fit_line(x_vector, y_vector):
    # 利用最小二乘法计算拟合的直线
    linreg = LinearRegression()
    x_vector = list(map(lambda _x: [_x], x_vector))
    y_vector = list(map(lambda _x: [_x], y_vector))

    linreg.fit(x_vector, y_vector)
    x = [[0], [0.1], [0.5], [0.9], [1.0]]
    y = linreg.predict(x)
    return x, y, float(linreg.coef_[0][0]), float(linreg.intercept_[0])


if __name__ == '__main__':
    base_path = './delect_node_effect/new/'
    network_name = 'ER'
    open_file_list = [i for i in os.listdir(base_path) if network_name in i]
    for file_name in open_file_list:
        print(file_name)

    with open(base_path + 'FVS_' + network_name + '_fomula.txt', 'r') as f:
        FVS = f.readlines()
    with open(base_path + 'LCC_' + network_name + '_fomula.txt', 'r') as f:
        LCC = f.readlines()
    # with open(base_path + 'LCCs_'+network_name + '_formula_2.txt', 'r') as f:
    #     LCCs = f.readlines()
    # with open(base_path + 'LCCs_s_'+network_name + '_formula_2.txt', 'r') as f:
    #     LCCs_s = f.readlines()
    with open(base_path + 's_' + network_name + '_fomula.txt', 'r') as f:
        s = f.readlines()

    FVS = list(map(lambda x: float(x.strip().split(' ')[0]), FVS[1:]))
    # LCCs = list(map(lambda x: float(x.strip().split(' ')[0]), LCCs[1:]))
    LCC = list(map(lambda x: float(x.strip().split(' ')[0]), LCC[1:]))
    # LCCs_s = list(map(lambda x: float(x.strip().split(' ')[0]), LCCs_s[1:]))
    s = list(map(lambda x: float(x.strip().split(' ')[0]), s[1:]))

    args = ([list(range(len(FVS))), FVS, 'BPD', '.', '5', '-', 'coral'],
            [list(range(len(LCC))), LCC, 'LCC', '.', '5', '-', 'yellowgreen'],
            # [list(range(len(LCCs))),LCCs,'LCCs','.','5','-','slateblue'],
            # [list(range(len(LCCs_s))),LCCs_s,'LCCs_s','.','5','-','hotpink'],
            [list(range(len(s))), s, 's', '.', '5', '-', 'gold'])
    # args = ([list(range(len(FVS))), FVS, 'BPD'], [list(range(len(s))), s, 's'],
    # [list(range(len(LCCs_s))),LCCs_s,'LCCs_s'])
    draw_line_chart(base_path + 'delect_node_' + network_name + '_chn20_new.png', network_name + u'网络扩散影响', u'删除节点数量',
                    u'扩散影响', *args)
