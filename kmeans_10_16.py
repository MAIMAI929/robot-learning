# 1. 导入所需库
import pandas as pd  #处理表格数据，可以快速读取、查看、筛选数据
import numpy as np    #处理数字计算
import matplotlib.pyplot as plt   #将数据变为散点图，折线图
from sklearn.cluster import KMeans   #kmeans工具
from sklearn.preprocessing import StandardScaler     #数据标准化工具，让不同单位的数据能放在一起


# 设置随机种子，保证结果可复现
np.random.seed(42)    #固定生成随机数的起点，确保结果可以复现

# 生成3个聚类簇的核心数据
# 簇1：低收入低消费（年收入2-5万，消费评分10-40）
cluster1 = np.random.normal(loc=[3.5, 25], scale=[1, 8], size=(50, 2))
# 簇2：中等收入中等消费（年收入7-10万，消费评分40-70）
cluster2 = np.random.normal(loc=[8.5, 55], scale=[1, 8], size=(50, 2))
# 簇3：高收入高消费（年收入12-15万，消费评分70-100）
cluster3 = np.random.normal(loc=[13.5, 85], scale=[1, 8], size=(50, 2))

# 合并数据并转换为DataFrame
data = np.vstack([cluster1, cluster2, cluster3])    #将多个数组按行的方向堆叠在一起
df = pd.DataFrame(data, columns=["Annual_Income", "Spending_Score"])  #将堆叠好的数据按列贴上标签，便于后面的数据筛选查找，类似于excel表格的列操作

# 确保数值为正数（符合业务逻辑）
df["Annual_Income"] = df["Annual_Income"].clip(lower=1, upper=15)   #clip用于限制数据的范围，低于下限修改为下限，超过上限改为上限
df["Spending_Score"] = df["Spending_Score"].clip(lower=1, upper=100).round(0).astype(int)  #clip用在这里可以消除一些不符合逻辑的异常值

# 导出为CSV文件
df.to_csv("kmeans_customer_data.csv", index=False, encoding="utf-8")  #index时索引值，改为False后不添加索引值
print("数据集已生成：kmeans_customer_data.csv")
print(f"数据量：{len(df)} 条")
print(f"前5条数据：\n{df.head()}")

# 设置中文字体，避免图表中文乱码
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 加载并查看数据集
# 读取CSV文件（确保文件路径与代码运行路径一致）
df = pd.read_csv("kmeans_customer_data.csv")
# 查看数据基本信息
print("数据集基本信息：")
print(f"数据形状：{df.shape}")  # 输出(150,2)，150条样本、2个特征
print("\n前5条数据：")
print(df.head())
print("\n数据描述性统计：")
print(df.describe())

# 3. 数据预处理（标准化）
# 提取特征列（年收入和消费评分）
X = df[["Annual_Income", "Spending_Score"]]
# 初始化标准化器，消除量纲影响（也就是消除了单位的影响使两个标签的数据都变成单纯的数值）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   #fit是让机器学习特征的统计信息，transform是为了让数据标准化，这里的fit_transform是两个函数办法fit 和transform的快捷拟合，等价于先调用fit，后调用transform
# 转换为DataFrame便于后续操作
X_scaled_df = pd.DataFrame(X_scaled, columns=["Scaled_Income", "Scaled_Score"])

# 4. 肘部法则（Elbow Method）选择最优聚类数K
# 初始化存储惯性值（簇内平方和）的列表
inertia = []
# 测试K从1到10的聚类效果
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init=10确保结果稳定
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # 记录每个K对应的惯性值     这里的kmeans.inertia可以直接使用是因为inertia是KMeans聚类方法在处理数据是自己生成的属性，用来记录簇内平方和即聚散程度

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.xlabel('聚类数量 K', fontsize=12)
plt.ylabel('惯性值（簇内平方和）', fontsize=12)
plt.title('肘部法则图：选择最优K值', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
# 标记肘部位置（此处K=3时惯性值下降幅度明显减小）
plt.annotate('肘部位置（K=3）', xy=(3, inertia[2]), xytext=(4, inertia[2]+5),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=11)
# 添加坐标轴含义说明
plt.figtext(0.5, 0.01, "横坐标：尝试的聚类数量K值；纵坐标：惯性值（衡量簇内样本的紧凑程度，值越小说明簇内样本越集中）",
            ha="center", fontsize=10, style='italic')
plt.savefig("elbow_method.png", dpi=300, bbox_inches='tight')  # 保存图片
plt.show()
print("\n肘部法则图已保存为 elbow_method.png")

# 5. 用最优K=3进行K-Means聚类训练
kmeans_optimal = KMeans(n_clusters=3, random_state=42, n_init=10)
# 在标准化数据上训练模型
kmeans_optimal.fit(X_scaled)
# 将聚类标签添加到原始数据集
df["Cluster_Label"] = kmeans_optimal.labels_
# 查看各簇的样本数量
print("\n各聚类簇的样本数量：")
print(df["Cluster_Label"].value_counts().sort_index())

# 6. 聚类结果可视化（原始特征空间）
plt.figure(figsize=(10, 6))
# 定义3个簇的颜色和标记
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']
# 遍历每个簇，绘制散点图
for cluster in range(3):
    cluster_data = df[df["Cluster_Label"] == cluster]
    plt.scatter(
        cluster_data["Annual_Income"],
        cluster_data["Spending_Score"],
        c=colors[cluster],
        marker=markers[cluster],
        s=80,
        label=f'簇 {cluster+1}',
        alpha=0.7
    )

# 添加图表标签和标题
plt.xlabel('年收入（万元）', fontsize=12)
plt.ylabel('消费评分（分）', fontsize=12)
plt.title('K-Means聚类结果（K=3）：客户消费行为分群', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
# 添加坐标轴含义详细说明
plt.figtext(0.5, 0.01, "横坐标：客户的年收入（单位：万元），范围1-15万元；纵坐标：客户的消费评分（单位：分），范围1-100分，分数越高表示消费意愿越强",
            ha="center", fontsize=10, style='italic')
# 保存聚类结果图
plt.savefig("kmeans_clustering_result.png", dpi=300, bbox_inches='tight')
plt.show()
print("聚类结果图已保存为 kmeans_clustering_result.png")

# 7. 输出各簇的特征统计（分析客户分群特征）
print("\n各聚类簇的特征统计（原始数据）：")
cluster_stats = df.groupby("Cluster_Label")[["Annual_Income", "Spending_Score"]].agg(['mean', 'std'])
print(cluster_stats.round(2))
# 保存统计结果到CSV
cluster_stats.round(2).to_csv("cluster_statistics.csv", encoding="utf-8")
print("\n各簇统计结果已保存为 cluster_statistics.csv")