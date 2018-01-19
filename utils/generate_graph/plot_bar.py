import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(15, 7.5))
n = 10
X = np.arange(n)+1

Y1 = [ 0.6875      ,0.72083342  ,0.67083335 , 0.90833342 , 0.70416671 , 0.81666672 ,0.78750002 , 0.72916675 , 0.81250012 , 0.77083337]
Y2 = [ 0.76666671 , 0.7458334 ,  0.74583334 , 0.86666667 , 0.64583337 , 0.8708334 ,0.85833329 , 0.73333335 , 0.8416667   ,0.84583342]
# Y3 = [ 0.71249998, 0.76458335, 0.66874999, 0.78750002, 0.59583342, 0.64791667, 0.76041669, 0.70000005, 0.68958336, 0.70416671]
plt.bar(X-0.15, Y1, width=0.3, facecolor='sandybrown', edgecolor='white', label='baseline:average=0.74')
plt.bar(X+0.15, Y2, width=0.3, facecolor='darksalmon', edgecolor='white', label='our network:average=0.77')
# plt.bar(X+0.3, Y3, width=0.3, facecolor='tan', edgecolor='white', label='(5:5):average=0.7031')
#width:柱的宽度
# plt.bar(X+0.35,Y2,width = 0.35,facecolor = 'yellowgreen',edgecolor = 'white')
#水平柱状图plt.barh，属性中宽度width变成了高度height
#打两组数据时用+
#facecolor柱状图里填充的颜色
#edgecolor是边框的颜色
#想把一组数据打到下边，在数据前使用负号
#plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
#给图加text
for x, y in zip(X, Y1):
    plt.text(x-0.15, y, str(round(y, 2)), ha='center', va='bottom', fontsize=12)

for x, y in zip(X,Y2):
    plt.text(x+0.15, y, str(round(y, 2)), ha='center', va= 'bottom', fontsize=12)

# for x, y in zip(X,Y3):
#     plt.text(x+0.3, y, str(round(y, 2)), ha='center', va= 'bottom')
# plt.ylim(0,+1.25)
plt.title('Result Compare', fontsize=25)
plt.ylabel('accuracy', fontsize=20)
plt.xlabel('fold number', fontsize=20)
plt.legend(loc=4, fontsize=20, shadow=True)
plt.savefig('F:\\files\\facial_expresssion\\graph\\test2.jpg', dpi=300, bbox_inches='tight')
plt.show()