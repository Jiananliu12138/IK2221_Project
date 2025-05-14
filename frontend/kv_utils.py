import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def aggregate_kv_tensor(kv_tensor, method="mean"):
    """
    将KV张量聚合为定长向量。
    kv_tensor: numpy数组，shape = (num_layers, num_heads, seq_len, head_dim)
    method: "mean" 或 "last_layer"
    返回: 1D向量
    """
    if method == "mean":
        # 对所有层、头、token做平均
        return kv_tensor.mean(axis=(0,1,2))
    elif method == "last_layer":
        # 只用最后一层
        return kv_tensor[-1].mean(axis=(0,1))
    else:
        raise ValueError("Unknown aggregation method")

def reduce_dim(vectors, n_components=64):
    """
    用PCA降维
    vectors: 2D numpy数组，每行一个样本
    返回: 降维后的2D数组
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)

def cluster_vectors(vectors, n_clusters):
    """
    KMeans聚类
    vectors: 2D numpy数组
    n_clusters: 聚类数
    返回: 聚类标签
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(vectors)

def evaluate_clustering(true_labels, cluster_labels):
    """
    评估聚类效果
    返回: ARI, NMI
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return ari, nmi

def kv_clustering_pipeline(kv_tensors, true_labels, n_clusters, agg_method="mean", n_components=64):
    """
    一步到位：聚合、降维、聚类、评估
    kv_tensors: List[np.ndarray]，每个元素是一个请求的KV张量
    true_labels: List[str/int]，每个请求的真实标签
    n_clusters: 聚类数
    agg_method: 聚合方法
    n_components: 降维维度
    返回: ARI, NMI, 聚类标签
    """
    agg_vectors = np.stack([aggregate_kv_tensor(kv, method=agg_method) for kv in kv_tensors])
    reduced_vectors = reduce_dim(agg_vectors, n_components=n_components)
    cluster_labels = cluster_vectors(reduced_vectors, n_clusters)
    ari, nmi = evaluate_clustering(true_labels, cluster_labels)
    return ari, nmi, cluster_labels


# import kv_utils

# # 假设你已经有如下数据
# # kv_tensors: List[np.ndarray]，每个请求的KV张量
# # true_labels: List[str/int]，每个请求的真实paper_id
# # n_clusters: 文档数

# ari, nmi, cluster_labels = kv_utils.kv_clustering_pipeline(
#     kv_tensors, true_labels, n_clusters=len(set(true_labels)), agg_method="mean", n_components=64
# )
# st.write(f"KV聚类 ARI: {ari:.3f}, NMI: {nmi:.3f}")