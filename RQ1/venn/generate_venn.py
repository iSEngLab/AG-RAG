import venn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_excel("./baselines_new.xlsx")
table = df.to_numpy()

list = [set(), set(), set(), set(), set()]

for row in table:
    for i in range(1, 6):
        if row[i] == 1:
            list[i - 1].add(row[0])

labels = venn.get_labels(list)
venn.venn5(labels, names=['ATLAS', 'EDITAS', 'Integration', 'IR', 'Ours'], dpi=300)

# 获取当前图形
fig = plt.gcf()

# 遍历所有文本对象
for text_obj in fig.findobj(matplotlib.text.Text):
    text_obj.set_fontsize(23)

plt.savefig('new.pdf', dpi=300, bbox_inches='tight')
plt.show()
