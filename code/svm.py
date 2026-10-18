import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ==================== 1. 构建数据集 ====================
data = Dataset("./celeba")
X_imgs_train, X_attrs_train, Y_train = data.get_train_data()
X_imgs_test, X_attrs_test, Y_test = data.get_test_data()

# --- 新增：加载并切分 Landmarks 数据 (利用celeba_landmarks的10个特征将40个属性扩展到60）---
# 1. 读取 landmarks 文件，存入字典 (跳过表头)
landmarks_dict = {}
with open("./celeba/celeba_landmarks.txt", "r") as f:
    header = f.readline() # 跳过表头
    for line in f:
        parts = line.strip().split()
        if not parts: continue
        img_name = parts[0]  # 获取文件名，例如 '125415.jpg'
        coords = [float(x) for x in parts[1:]] 
        landmarks_dict[img_name] = coords

# 2. 利用 Dataset 提取出的索引，获取绝对对应顺序的文件名列表
train_filenames = [data.filenames[i] for i in data.train_indices]
test_filenames = [data.filenames[i] for i in data.test_indices]

# 3. 按照正确的文件名顺序提取关键点特征
X_landmarks_train = np.array([landmarks_dict[name] for name in train_filenames])
X_landmarks_test = np.array([landmarks_dict[name] for name in test_filenames])

# ==================== 2. 选择分类依据（特征） ====================
# FEATURE = 'attr'  -> 用 40 维人脸属性作为特征（40 维）
# FEATURE = 'image' -> 用原始图片像素作为特征（178*218*3 = 116412 维）
# FEATURE = 'fusion' -> 增加了20维关键点
FEATURE = 'fusion'

if FEATURE == 'image':
    # 选择原始图片作为分类依据
    # 单张图片的原始形状是 [H, W, 3]，而 SVM 的输入必须是一个一维向量，
    # 所以要把后三个维度展平： (N, H, W, 3) -> (N, H*W*3)
    # reshape 的第 2 个参数写 -1 表示"这一维由 numpy 自己算出来"，避免手写 178*218*3
    X_train = X_imgs_train.reshape(X_imgs_train.shape[0], -1)
    X_test = X_imgs_test.reshape(X_imgs_test.shape[0], -1)
elif FEATURE == 'attr':
    # 选择属性特征作为分类依据，形状 (N, 40)
    X_train = X_attrs_train
    X_test = X_attrs_test
elif FEATURE == 'fusion':
    # --- 新增：融合特征 ---
    # 使用 np.hstack 将 40维属性 和 20维关键点 横向拼接
    X_train = np.hstack((X_attrs_train, X_landmarks_train))
    X_test = np.hstack((X_attrs_test, X_landmarks_test))

print('特征: %s, train=%s, test=%s' % (FEATURE, X_train.shape, X_test.shape))

# ==================== 3. 标准化训练集和测试集 ====================
sc = StandardScaler()               # 定义一个标准缩放器
sc.fit(X_train)                     # 计算均值、标准差（只能用训练集算）
X_train_std = sc.transform(X_train) # 使用计算出的均值和标准差进行标准化
X_test_std  = sc.transform(X_test)  # 测试集复用训练集的均值和标准差, 避免信息泄露

# ==================== 4. 训练支持向量机 ====================
# 10 个类别不需要手写多个 SVM：SVC 内部按 one-vs-one 拆成 C(10,2)=45 个二分类器
# 每个只用两类的样本训练, 预测时 45 个分类器投票, 票数最多者胜。
svm = SVC(
    kernel='linear',   # 核函数: 线性核 K(x, z) = x·z
    C=1.0,             # 软间隔惩罚系数: C 越大越不能容忍错分(间隔越窄), C 越小正则越强
)
svm.fit(X_train_std, Y_train)       # 根据给定的训练数据拟合训练SVM模型

# ==================== 5. 使用测试集进行数据预测 ====================
Y_pred = svm.predict(X_test_std)    # 用训练好的分类器svm预测数据X_test_std的标签
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())   # 输出错误分类的样本数
print('Accuracy: %.2f' % svm.score(X_test_std, Y_test))         # 输出分类准确率

# ==================== 6. 绘制混淆矩阵 ====================
# 1. 计算混淆矩阵
# cm 是一个 N x N 的矩阵（N是类别数，你这里是10）
# 行代表真实类别，列代表预测类别
cm = confusion_matrix(Y_test, Y_pred)

# 2. 可视化混淆矩阵
class_names = ['Person 0', 'Person 1', 'Person 2', 'Person 3', 'Person 4',
               'Person 5', 'Person 6', 'Person 7', 'Person 8', 'Person 9']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# cmap=plt.cm.Blues 表示使用蓝色色系，颜色越深代表数量越多
disp.plot(cmap=plt.cm.Blues, values_format='d') 
# 标签倾斜一下，防止重叠
plt.xticks(rotation=45) 
# 添加标题
plt.title('Confusion Matrix (Feature: attr)')

# 显示图像
plt.show()