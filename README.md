## jupyter快捷键
- `Enter`切换为编辑模式
- Esc`切换为命令模式

### 命令模式快捷键
- `H`：显示快捷键帮助
- `F`：查找和替换
- `P`：打开命令面板
- `Alt-Enter`：运行当前cell并在下方新建cell
- `Y`：把当前cell内容转换为代码形式
- `M`：把当前cell内容转换为markdown形式
- `1~6`：把当前cell内容设置为标题1~6格式
- `Shift`+上下键：按住Shift进行上下键操作可复选多个cell
- `A`：在上方新建cell
- `B`：在下方新建cell
- `X/C/Shift-V/V`：剪切/复制/上方粘贴/下方粘贴
- 双击`D`：删除当前cell
- `Z`：撤销删除
- `S`：保存notebook
- `L`：为当前cell的代码添加行编号
- `Shift-L`：为所有cell的代码添加行编号
- `Shift-M`：合并所选cell或合并当前cell和下方的cell
- 双击`I`：停止kernel
- 双击`0`：重启kernel
- 在一个库，方法或变量前加上 ?，你可以获得它的一个快速语法说明。

### 编辑模式快捷键
- `Tab`：代码补全
- `Shift-Tab` : 提示
- `Ctrl-A`：全选
- `Ctrl-Z`：撤销
- `Ctrl-Home`：将光标移至cell最前端
- `Ctrl-End`：将光标移至cell末端
- `Ctrl-/`：单行或多行注释与取消
- `Ctrl-]`: 缩进
- `Ctrl-[` : 解除缩进
- `jupyter notebook`误删cell 之后不要关闭窗口，也不要停止运行，直接在命令行按下`z`键就可以了。还有一个方式就是万能的 `history`，命令输入shift + 回车

## scikit-learn
- `fit`：从一个训练集中学习模型参数，包括归一化时用到的均值，标准偏差；
- `transform`：将模型用于位置数据；
- `fit_transform`：将模型训练和转化合并到一起；
- 训练样本先做fit，得到mean，standard deviation，然后将这些参数用于transform（归一化训练数据），使得到的训练数据是归一化的，而测试数据只需要在原先得到的mean，std上做归一化。
- `StandardScaler`对矩阵作归一化处理，变换后的矩阵各特征均值为0，方差为1。
- `sklearn `提供了两种通用的参数搜索/采样方法：网络搜索和随机采样，
  - 网格搜索交叉验证（`GridSearchCV`）：以穷举的方式遍历所有可能的参数组合
  - 随机采样交叉验证（`RandomizedSearchCV`）：依据某种分布对参数空间采样，随机的得到一些候选参数组合方案
- `np.random.shuffle(x)`：现场修改序列，改变自身内容。（类似洗牌，打乱顺序）对多维数组进行打乱排列时，默认是对第一个维度也就是列维度进行随机打乱
- `np.random.permutation(x)`： 返回一个随机排列，如果x是一个多维数组，则只会沿着它的第一个索引进行随机排列。
  - axis=0：表示输出矩阵是1行，即求每一列的平均值。
  - axis=1：表示输出矩阵是1列, 即求每一行的平均值。
  - 实际上axis=0就是选择shape中第一维变为1，axis=1选择shape中第二维变为1。

### clean data
- `%matplotlib inline`
  - import matplotlib.pyplot as plt
  - plt.rcParams['font.sans-serif'] = ['SimHei']
  - plt.rcParams['axes.unicode_minus'] = False

- `import warnings`
  - warnings.filterwarnings("ignore", message="numpy.dtype size changed")
  - warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

- `import os`
  - os.environ['path']
  - os.getenv('path')
  - os.getcwd()

- `import sys`
  - current_path=os.getcwd()
  - sys.path.append(current_path)`

- `pd.set_option('display.max_columns', None)`显示所有列
- `pd.set_option('display.max_rows', None)`显示所有行
- `pd.set_option('max_colwidth',100)`设置value的显示长度为100，默认为50
- `df.columns = [str(col) + '_x' for col in df.columns]`列名加前后缀

- 用中括号[a,b]表示样本估计总体平均值误差范围的区间，a、b的具体数值取决于你对于”该区间包含总体均值”这一结果的可信程度，[a,b]被称为置信区间。最常用的95%置信水平，就是说做100次抽样，有95次的置信区间包含了总体均值。
- 分类交叉熵是分类问题合适的损失函数，它最小化模型输出的概率分布和真实label的概率分布之间的距离。
- 处理多分类中label的两种方法：
  - 通过one-hot编码编码label，并使用categorical_crossentropy作为损失函数；
  - 通过整数张量编码label，并使用sparse_categorical_crossentropy损失函数，对于数据分类的类别较多的情况，应该避免创建较小的中间layer，导致信息瓶颈。
- 常见回归指标是平均绝对误差（MAE）,若可用数据较少，使用K折验证进行可靠评估。
- `roc_auc_score()`：首先获得roc曲线,然后调用auc()来获取该区域。调用`predict_proba()`，对于正常的预测,输出总是相同的。传入预测的分类结果和预测的概率都是可以计算的。正确的做法是传入预测概率，这样才符合AUC的计算原理。并且传入分类结果的话，AUC指标会更低，因为曲线变粗糙了。
- `Pearson`相关系数要求两个连续性变量符合正态分布,不服从正态分布的变量、分类或等级变量之间的关联性可采用`Spearman`秩相关系数，也称为等级相关系数。
