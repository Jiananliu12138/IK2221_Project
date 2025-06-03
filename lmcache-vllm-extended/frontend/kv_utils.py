import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def aggregate_kv_tensor(kv_tensor, method="mean"):
    """
    KV张量聚合
    kv_tensor: numpy数组 shape = (num_layers, num_heads, seq_len, head_dim)
    method: "mean" 
    返回: 1D向量
    """
    key_tensors = [k[0].cpu().numpy() for k in kv_tensor]  # 28个(1, 2, 709, 128)
    key_tensors = np.stack(key_tensors) #(28, 1, 2, 709, 128)
    if method == "mean":
        # 对所有层、头、token做平均
        agg = key_tensors.mean(axis=(1,2,3))  # (num_layers, head_dim) (28, 128)
        return agg.flatten() #展平成一维（28×128=3584）

def reduce_dim(vectors):
    """
    用PCA降维
    vectors: 2D numpy数组每行一个样本
    返回: 降维后的2D数组
    """
    n_samples, n_features = vectors.shape
    n_components = min(64, n_samples, n_features)
    # print(f"Reducing dimensions from {n_samples}和{n_features} to {n_components}")
    # Reducing dimensions from 9和3584 to 9
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)#(9,3584)=>(9,9)

def cluster_vectors(vectors, n_clusters):
    """
    KMeans聚类
    vectors: 2D numpy数组
    n_clusters: 聚类数
    返回: 聚类标签
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(vectors)#(9,1)

def evaluate_clustering(true_labels, cluster_labels):
    """
    评估聚类效果
    返回: ARI, NMI
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return ari, nmi

def kv_clustering_pipeline(kv_tensors, true_labels, n_clusters, agg_method="mean"):
    """
    聚合、降维、聚类、评估
    kv_tensors: 每个元素是一个请求的KV张量
    true_labels: 每个请求的真实标签
    n_clusters: 聚类数
    agg_method: 聚合方法
    n_components: 降维维度
    返回: ARI, NMI, 聚类标签
    """
    agg_vectors = np.stack([aggregate_kv_tensor(kv, method=agg_method) for kv in kv_tensors])#(9, 3584)
    reduced_vectors = reduce_dim(agg_vectors)
    cluster_labels = cluster_vectors(reduced_vectors, n_clusters)
    print(f"聚类标签: {cluster_labels} 真实标签: {true_labels}")
    ari, nmi = evaluate_clustering(true_labels, cluster_labels)
    return ari, nmi, cluster_labels
