import numpy as np
import pandas as pd
from collections import Counter

# 生成所需的数据
df = pd.DataFrame({
    'x1': pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
    'x2': pd.Series(['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']),
    'y': pd.Series([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]),
})

prov = {}
x = (2, 'S')
# 使用极大似然估计计算先验概率和条件概率
c = Counter(df['y'])
prov[1] = c[1] / df['y'].shape[0]
prov[-1] = c[-1] / df['y'].shape[0]
for key in dict(prov):
    # 计算后验概率
    a1 = np.sum(np.sum(df.loc[:, ['x1', 'y']] == [x[0], key], axis=1) == 2) / c[key]
    a2 = np.sum(np.sum(df.loc[:, ['x2', 'y']] == [x[1], key], axis=1) == 2) / c[key]
    prov[key] = prov[key] * a1 * a2
ans = 0
val = 0
for key in dict(prov):
    if prov[key] > val:
        ans = key
        val = prov[key]
print(ans)