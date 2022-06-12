# Numpy

Numpy 的数据结构称为 ndarray，相当于 python 的数组，可以看成是在数组的基础上增加了许多用于数据分析方面的操作。

> 使 Numpy 更快的方法：https://mofanpy.com/tutorials/data-manipulation/np-pd/speed-up-numpy/

## ndarray的属性

```python
ndarray = np.array([[1,2,3],[2,3,4]]) # 将python数组转化为numpy的数组类型ndarray
print(ndarray)
"""
array([[1, 2, 3],
       [2, 3, 4]])
"""
print('number of dim:', ndarray.ndim) # 数组的维度, 这里是二维数组 (表示向量或矩阵)
# number of dim: 2
print('shape:', ndarray.shape) # 数组的形状, 即数组在各个维度上的长度
# shape: (2, 3)
print('size:', ndarray.size) # 数组的元素个数
# size: 6

# 零维数组表示一个数值, 形状为空, 即 ()
# 一维数组表示列表, 形状为 (元素个数,)
# 二维数组表示向量或矩阵, 形状为 (行数, 列数)
```

![4-1-3.png](https://static.mofanpy.com/results/np-pd/4-1-3.png)

## 创建ndarray的方法

```python
# 根据传入的数组创建ndarray
a = np.array([2, 23, 4], dtype=np.int)
# 根据传入的形状创建值全0的ndarray
a = np.zeros((3, 4))
# 根据传入的形状创建值全1的ndarray
a = np.ones((3, 4), dtype=np.int)
# 根据传入的形状创建值接近于0的ndarray
a = np.empty((3, 4))
# 创建起始为10, 终值为20, 步长为2的一维序列
a = np.arange(10, 20, 2)
a = np.arange(12).reshape((3, 4))
# 创建起始为1, 终值为10, 元素个数为20的一维序列
a = np.linspace(1, 10, 20)
a = np.linspace(1, 10, 20).reshape((5, 4))
# 根据传入的形状创建值随机的ndarray
a=np.random.random((2,4))
```

## ndarray的运算

```python
# 逐元素计算的运算
c = a-b
c = a*b
c = b**2
c = 10*np.sin(a)
b < 3
b == 2

# 矩阵运算
c = np.dot(a,b)
c = a.dot(b)

# 返一运算
np.sum(a)
np.min(a)
np.max(a)
np.argmin(a)
np.argmax(a)
np.mean(a)
np.average(a)
np.median(a)

# 按轴运算
# 凡是有 axis 参数的运算, 都可以变成按轴运算
np.sum(a, axis=1) # 按列排列的方向进行求和, 即将每一行进行求和
np.min(a, axis=0) # 按行排列的方向进行求最小值, 即求每一列的最小值
np.max(a, axis=1) # 按列排列的方向进行求最大值
np.sort(a) # 逐行排序

# 特殊运算
np.cumsum(a) # 返回累加序列
np.diff(a) # 返回累差序列
np.nonzero(a) # 返回非零元素在各个维度上的索引
np.transpose(a) # 矩阵转置, 同 a.T
np.clip(A,5,9) # 截取运算, 令小于5的元素等于5, 大于9的元素等于9, 中间的元素保持不变
a.flatten() # 将数组展成一维数组
```

## ndarray的索引

```python
# numpy数组和python数组对于索引的使用完全一致
A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
print(A[2]) # [11 12 13 14]
print(A[1][1]) # 8
print(A[1, 1]) # 8
# 同样可以使用切片操作
print(A[2, :]) # [11 12 13 14]
print(A[1, 1:3]) # [8 9]
print(A[:, 0]) # [3 7 11]
# 可以进行for循环
for row in A: print(row)
for column in A.T: print(column)
for item in A.flat: print(item)
```

## ndarray的合并

```python
A = np.array([1,1,1])
B = np.array([2,2,2])

# 在垂直方向上合并 (就像两个矩阵进行拼接)
print(np.vstack((A,B)))
"""
[[1,1,1]
 [2,2,2]]
"""
# 在水平方向上合并
print(np.hstack((A,B))) # [1,1,1,2,2,2]
# 多数组合并
C = np.concatenate((A,B,B,A), axis=0) # axis=0 表示在行排列的方向(垂直方向)上进行合并
D = np.concatenate((A,B,B,A), axis=1) # axis=1 表示在列排列的方向(水平方向)上进行合并

# 增加维度
print(A[np.newaxis,:]) # [[1 1 1]]
print(A[:,np.newaxis])
"""
[[1]
 [1]
 [1]]
"""
```

## ndarray的分割

```python
A = np.arange(12).reshape((3, 4))
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""

# 将 A 在水平方向上等量分割成 2 个
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
# 将 A 在垂直方向上等量分割成 3 个
print(np.split(A, 3, axis=0))
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
# NOTE: np.split 只能进行等量对分

# 将 A 在水平方向上不等量分割成 3 个
print(np.array_split(A, 3, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2],
        [ 6],
        [10]]), array([[ 3],
        [ 7],
        [11]])]
"""

# 其他方法
print(np.vsplit(A, 3)) # 等于 print(np.split(A, 3, axis=0))
print(np.hsplit(A, 2)) #等于 print(np.split(A, 2, axis=1))
```

## ndarray的拷贝

```python
a = np.arange(4) # array([0, 1, 2, 3])

# ndarray的赋值传递的是引用, 即浅拷贝
b = a
d = b
b is a  # True
d is a  # True
d[1:3] = [22, 33]   # array([0, 22, 33,  3])
print(a)            # array([0, 22, 33,  3])
print(b)            # array([0, 22, 33,  3])

# 深拷贝
b = a.copy()
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])
```

# Pandas

Numpy 的数据结构是 ndarray，相当于 python 里的数组。与 Pandas 不同的是，在 ndarray 中的元素没有别名（或是标签）。也就是说，我们想要引用或获取 ndarray 里的元素，只能使用索引，而无法使用一个具体的名称。而 Pandas 的数据结构主要有 Series 和 DataFrame，相当于 python 里的字典。也就是说，我们可以给 Series 和 DataFrame 里的元素定义别名，然后用别名去引用它们。

```python
# Series 相当于一维数组, 不过该"数组"的索引 (在 Pandas 中称为标签) 可以是数字，也可以是具体的名称
# 如果不指定名称，则默认采用 0-n 的数值作为标签
# 创建 Series
print(pd.Series([1,3,6,np.nan,44,1]))
"""
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
"""

# 产生日期序列
dates = pd.date_range('20160101',periods=6)
# 创建 DataFrame, 将上面产生的日期序列作为行标签, 而列标签为 a, b, c, d
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c','d'])
print(df)
"""
                   a         b         c         d
2016-01-01 -0.253065 -2.071051 -0.640515  0.613663
2016-01-02 -1.147178  1.532470  0.989255 -0.499761
2016-01-03  1.221656 -2.390171  1.862914  0.778070
2016-01-04  1.473877 -0.046419  0.610046  0.204672
2016-01-05 -1.584752 -0.700592  1.487264 -1.778293
2016-01-06  0.633675 -1.414157 -0.277066 -0.442545
"""

# 创建 DataFrame, 不指定标签, 则默认采用数值作为标签
df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

"""
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
"""

# 根据传入的字典来创建 DataFrame, 字典的键就是列标签, 字典的值就是每列的内容
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
                    
print(df2)

"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```

## 一些基本操作

```python
print(df2.dtypes) # 打印类型
"""
df2.dtypes
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
"""

print(df2.index) # 打印行标签
# Int64Index([0, 1, 2, 3], dtype='int64')

print(df2.columns) # 打印列标签
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')

print(df2.values) # 打印各元素的值
"""
array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo']], dtype=object)
"""

df2.describe() # 返回各列元素的一些属性, 比如元素数量、均值、最小值等
"""
         A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
"""

print(df2.T) # 转置
"""                   
0                    1                    2  \
A                    1                    1                    1   
B  2013-01-02 00:00:00  2013-01-02 00:00:00  2013-01-02 00:00:00   
C                    1                    1                    1   
D                    3                    3                    3   
E                 test                train                 test   
F                  foo                  foo                  foo   

                     3  
A                    1  
B  2013-01-02 00:00:00  
C                    1  
D                    3  
E                train  
F                  foo  

"""

print(df2.sort_index(axis=1, ascending=False)) # 在列排列方向 (水平方向) 上降序排列
"""
     F      E  D    C          B    A
0  foo   test  3  1.0 2013-01-02  1.0
1  foo  train  3  1.0 2013-01-02  1.0
2  foo   test  3  1.0 2013-01-02  1.0
3  foo  train  3  1.0 2013-01-02  1.0
"""

print(df2.sort_values(by='B')) # 对标签"B"所在列的内容进行排序
"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```

## Pandas选择数据

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""

# 根据下标、对象属性、切片的方式选取元素
print(df['A']) # 等同于 print(df.A)
"""
2013-01-01     0
2013-01-02     4
2013-01-03     8
2013-01-04    12
2013-01-05    16
2013-01-06    20
Freq: D, Name: A, dtype: int64
"""
print(df[0:3])
"""
            A  B   C   D
2013-01-01  0  1   2   3
2013-01-02  4  5   6   7
2013-01-03  8  9  10  11
"""
print(df['20130102':'20130104'])
"""
			 A   B   C   D
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
"""

# 根据标签选取值
print(df.loc['20130102'])
"""
A    4
B    5
C    6
D    7
Name: 2013-01-02 00:00:00, dtype: int64
"""
print(df.loc[:,['A','B']]) 
"""
             A   B
2013-01-01   0   1
2013-01-02   4   5
2013-01-03   8   9
2013-01-04  12  13
2013-01-05  16  17
2013-01-06  20  21
"""
print(df.loc['20130102',['A','B']])
"""
A    4
B    5
Name: 2013-01-02 00:00:00, dtype: int64
"""

# 根据位置选取值
print(df.iloc[3,1]) # 13
print(df.iloc[3:5,1:3])
"""
             B   C
2013-01-04  13  14
2013-01-05  17  18
"""
print(df.iloc[[1,3,5],1:3])
"""
             B   C
2013-01-02   5   6
2013-01-04  13  14
2013-01-06  21  22

"""
```

## Pandas设置值

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""

df.iloc[2,2] = 1111
df.loc['20130101','B'] = 2222
"""
             A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     9  1111  11
2013-01-04  12    13    14  15
2013-01-05  16    17    18  19
2013-01-06  20    21    22  23
"""

df.B[df.A>4] = 0
"""
                A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     0  1111  11
2013-01-04  12     0    14  15
2013-01-05  16     0    18  19
2013-01-06  20     0    22  23 
"""

df['F'] = np.nan
"""
             A     B     C   D   F
2013-01-01   0  2222     2   3 NaN
2013-01-02   4     5     6   7 NaN
2013-01-03   8     0  1111  11 NaN
2013-01-04  12     0    14  15 NaN
2013-01-05  16     0    18  19 NaN
2013-01-06  20     0    22  23 NaN
"""

df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101',periods=6)) 
"""
             A     B     C   D   F  E
2013-01-01   0  2222     2   3 NaN  1
2013-01-02   4     5     6   7 NaN  2
2013-01-03   8     0  1111  11 NaN  3
2013-01-04  12     0    14  15 NaN  4
2013-01-05  16     0    18  19 NaN  5
2013-01-06  20     0    22  23 NaN  6
"""
```

## 处理失效数据

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
"""
             A     B     C   D
2013-01-01   0   NaN   2.0   3
2013-01-02   4   5.0   NaN   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""

# 使用函数 df.dropna 处理存在 NaN 的数据
df.dropna(
    axis=0,     # 0: 向下扫描, 按行删除; 1: 向右扫描, 按列删除
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 掉
) 
"""
             A     B     C   D
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""

# 使用函数 df.fillna 处理存在 NaN 的数据
df.fillna(value=0) # 用数值 0 代替 NaN
"""
             A     B     C   D
2013-01-01   0   0.0   2.0   3
2013-01-02   4   5.0   0.0   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""

# 判断各个元素是否是 NaN 数据
df.isnull() 
"""
                A      B      C      D
2013-01-01  False   True  False  False
2013-01-02  False  False   True  False
2013-01-03  False  False  False  False
2013-01-04  False  False  False  False
2013-01-05  False  False  False  False
2013-01-06  False  False  False  False
"""

# 判断是否存在 NaN 数据
np.any(df.isnull()) == True # True
```

## 导入和导出数据

Pandas 支持的读取和保存的数据格式：

| Format Type | Data Description                                             | Reader                                                       | Writer                                                       |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| text        | [CSV](https://en.wikipedia.org/wiki/Comma-separated_values)  | [read_csv](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-read-csv-table) | [to_csv](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-store-in-csv) |
| text        | Fixed-Width Text File                                        | [read_fwf](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-fwf-reader) |                                                              |
| text        | [JSON](https://www.json.org/)                                | [read_json](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-json-reader) | [to_json](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-json-writer) |
| text        | [HTML](https://en.wikipedia.org/wiki/HTML)                   | [read_html](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-read-html) | [to_html](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-html) |
| text        | Local clipboard                                              | [read_clipboard](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-clipboard) | [to_clipboard](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-clipboard) |
| binary      | [MS Excel](https://en.wikipedia.org/wiki/Microsoft_Excel)    | [read_excel](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-excel-reader) | [to_excel](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-excel-writer) |
| binary      | [OpenDocument](http://www.opendocumentformat.org/)           | [read_excel](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-ods) |                                                              |
| binary      | [HDF5 Format](https://support.hdfgroup.org/HDF5/whatishdf5.html) | [read_hdf](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-hdf5) | [to_hdf](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-hdf5) |
| binary      | [Feather Format](https://github.com/wesm/feather)            | [read_feather](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-feather) | [to_feather](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-feather) |
| binary      | [Parquet Format](https://parquet.apache.org/)                | [read_parquet](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-parquet) | [to_parquet](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-parquet) |
| binary      | [ORC Format](https://orc.apache.org/)                        | [read_orc](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-orc) |                                                              |
| binary      | [Msgpack](https://msgpack.org/index.html)                    | [read_msgpack](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-msgpack) | [to_msgpack](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-msgpack) |
| binary      | [Stata](https://en.wikipedia.org/wiki/Stata)                 | [read_stata](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-stata-reader) | [to_stata](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-stata-writer) |
| binary      | [SAS](https://en.wikipedia.org/wiki/SAS_(software))          | [read_sas](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-sas-reader) |                                                              |
| binary      | [SPSS](https://en.wikipedia.org/wiki/SPSS)                   | [read_spss](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-spss-reader) |                                                              |
| binary      | [Python Pickle Format](https://docs.python.org/3/library/pickle.html) | [read_pickle](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-pickle) | [to_pickle](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-pickle) |
| SQL         | [SQL](https://en.wikipedia.org/wiki/SQL)                     | [read_sql](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-sql) | [to_sql](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-sql) |
| SQL         | [Google BigQuery](https://en.wikipedia.org/wiki/BigQuery)    | [read_gbq](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-bigquery) | [to_gbq](https://pandas.pydata.org/docs/user_guide/io.html?highlight=read#io-bigquery) |

举例说明：

```python
# 读取 csv 格式的数据
data = pd.read_csv('example.csv')

# 保存成 pickle 格式的数据
data.to_pickle('example.pickle')
```

## Pandas合并

使用 concat 进行合并：

```python
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#concat纵向合并
res = pd.concat([df1, df2, df3], axis=0)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 0  1.0  1.0  1.0  1.0
# 1  1.0  1.0  1.0  1.0
# 2  1.0  1.0  1.0  1.0
# 0  2.0  2.0  2.0  2.0
# 1  2.0  2.0  2.0  2.0
# 2  2.0  2.0  2.0  2.0

# 将 index_ignore 设定为 True, 以重置行索引
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  2.0  2.0  2.0  2.0
# 7  2.0  2.0  2.0  2.0
# 8  2.0  2.0  2.0  2.0


df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

# 纵向"外"合并 (保留所有列标签, 用 NaN 填充缺失值) df1 与 df2
res = pd.concat([df1, df2], axis=0, join='outer')
#     a    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN
# 2  0.0  0.0  0.0  0.0  NaN
# 3  0.0  0.0  0.0  0.0  NaN
# 2  NaN  1.0  1.0  1.0  1.0
# 3  NaN  1.0  1.0  1.0  1.0
# 4  NaN  1.0  1.0  1.0  1.0

# 纵向"内"合并 (仅保留相同列标签的部分) df1 与 df2
res = pd.concat([df1, df2], axis=0, join='inner')
#     b    c    d
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  0.0  0.0  0.0
# 2  1.0  1.0  1.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0
```

使用 reindex 和 reindex_like 重置索引：

```python
# 使用 reindex 重置行索引
>>> df1.reindex([1, 2, 3, 4], fill_value=1)
#     a    b    c    d
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  0.0  0.0  0.0  0.0
# 4  1.0  1.0  1.0  1.0

# 使用 reindex_like 重置行索引
df1.reindex_like(df2)
#     a    b    c    d
# 2  0.0  0.0  0.0  0.0
# 3  0.0  0.0  0.0  0.0
# 4  NaN  NaN  NaN  NaN
```

使用 append 进行数据合并：

```python
# 使用 append 添加数据
res = df1.append(df2, ignore_index=True)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0

# 合并多个 df
res = df1.append([df2, df3], ignore_index=True)

# 合并 Series
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  2.0  3.0  4.0
```

使用 merge 进行数据合并：

```python
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
# left              right
#    A   B   key       C   D   key
# 0  A0  B0  K0 	0  C0  D0  K0
# 1  A1  B1  K1 	1  C1  D1  K1
# 2  A2  B2  K2 	2  C2  D2  K2
# 3  A3  B3  K3 	3  C3  D3  K3

# 依据某个 column 合并
res = pd.merge(left, right, on='key')
#    A   B   key C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K2  C2  D2
# 3  A3  B3  K3  C3  D3



left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
# left                      right
#    A   B key1 key2     		C   D key1 key2
# 0  A0  B0   K0   K0 		0  C0  D0   K0   K0
# 1  A1  B1   K0   K1  		1  C1  D1   K1   K0
# 2  A2  B2   K1   K0  		2  C2  D2   K1   K0
# 3  A3  B3   K2   K1  		3  C3  D3   K2   K0

# 依据两个 column 进行合并
# 'inner': 表示左右两个数据中 key1 和 key2 均相同的部分才会被合并
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
#    A   B key1 key2   C   D
# 0  A0  B0   K0   K0  C0  D0
# 1  A2  B2   K1   K0  C1  D1
# 2  A2  B2   K1   K0  C2  D2

# 'outer': 表示左右两个数据的 key1 和 key2 均保留, 用 NaN 填充缺失的数据
res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
#     A    B key1 key2    C    D
# 0   A0   B0   K0   K0   C0   D0
# 1   A1   B1   K0   K1  NaN  NaN
# 2   A2   B2   K1   K0   C1   D1
# 3   A2   B2   K1   K0   C2   D2
# 4   A3   B3   K2   K1  NaN  NaN
# 5  NaN  NaN   K2   K0   C3   D3

# 'left': 表示保留左边数据中 key1 和 key2 的值, 用 NaN 填充缺失的数据
res = pd.merge(left, right, on=['key1', 'key2'], how='left')
#    A   B key1 key2    C    D
# 0  A0  B0   K0   K0   C0   D0
# 1  A1  B1   K0   K1  NaN  NaN
# 2  A2  B2   K1   K0   C1   D1
# 3  A2  B2   K1   K0   C2   D2
# 4  A3  B3   K2   K1  NaN  NaN

# 'right': 表示保留右边数据中 key1 和 key2 的值, 用 NaN 填充缺失的数据
res = pd.merge(left, right, on=['key1', 'key2'], how='right')
#     A    B key1 key2   C   D
# 0   A0   B0   K0   K0  C0  D0
# 1   A2   B2   K1   K0  C1  D1
# 2   A2   B2   K1   K0  C2  D2
# 3  NaN  NaN   K2   K0  C3  D3



df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
# df1               	df2
#    col1 col_left  	    col1  col_right
# 0     0        a  	0     1          2
# 1     1        b  	1     2          2
#                   	2     2          2

# 依据 col1 进行合并, 并启用 indicator=True 将合并的记录放在新的一列
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
#   col1 col_left  col_right      _merge
# 0   0.0        a        NaN   left_only
# 1   1.0        b        2.0        both
# 2   2.0      NaN        2.0  right_only
# 3   2.0      NaN        2.0  right_only

# 自定义 indicator column 的名称
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
#   col1 col_left  col_right indicator_column
# 0   0.0        a        NaN        left_only
# 1   1.0        b        2.0             both
# 2   2.0      NaN        2.0       right_only
# 3   2.0      NaN        2.0       right_only



left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
# left        	right
#     A   B   	    C   D
# K0  A0  B0  	K0  C0  D0
# K1  A1  B1  	K2  C2  D2
# K2  A2  B2  	K3  C3  D3

# 依据左右数据的 index 进行合并, 并且 how='outer'
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
#      A    B    C    D
# K0   A0   B0   C0   D0
# K1   A1   B1  NaN  NaN
# K2   A2   B2   C2   D2
# K3  NaN  NaN   C3   D3

# 依据左右数据的 index 进行合并, 并且 how='inner'
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
#     A   B   C   D
# K0  A0  B0  C0  D0
# K2  A2  B2  C2  D2



boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
# 使用 suffixes 解决 overlapping 的问题,
# 即合并后重命名同名列标签, 使得可以区分同名列标签的从属关系
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
#    age_boy   k  age_girl
# 0        1  K0         4
# 1        1  K0         5
```

## 可视化数据

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Series 的可视化
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data.cumsum()
data.plot() # pandas 数据可以直接观看其可视化形式
plt.show()

# DataFrame的可视化
data = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
)
data.cumsum()
data.plot()
plt.show()

# 使用散点图进行可视化
axis = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
# ax=axis 表示画在 axis 所在的图上
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=axis)
plt.show()
```

