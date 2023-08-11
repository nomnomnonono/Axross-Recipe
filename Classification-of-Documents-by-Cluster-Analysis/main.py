import argparse
import glob
import os

import matplotlib.pyplot as plt
import MeCab
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data(num_files: int, categories: list[str]) -> tuple[list[str], list[str]]:
    texts, labels = [], []

    for category in categories:
        paths = sorted(glob.glob(os.path.join("text", category, f"{category}*.txt")))[
            :num_files
        ]
        for path in paths:
            with open(path) as f:
                texts.append("".join(f.readlines()[2:]))

            labels.append(os.path.splitext(os.path.basename(path))[0])

    return texts, labels


def wakati(texts: list[str]) -> list[str]:
    morpeme_texts = []
    mecab = MeCab.Tagger("-Owakati")

    for text in texts:
        morpeme_texts.append(mecab.parse(text))

    return morpeme_texts


def tfidf_vectorizer(morpeme_texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(morpeme_texts).toarray()

    return vector


def hierarchical_cluster(vector: np.ndarray, labels: list[str]) -> None:
    distance_matrix = pdist(vector, metric="cosine")

    clusters = linkage(distance_matrix, method="ward")

    plt.figure(figsize=(10, 5), dpi=150, tight_layout="True")
    dendrogram(clusters, labels=labels)
    plt.savefig("dendrogram.png")


def kmean_cluster(vector: np.ndarray, categories: list[str]) -> None:
    pca = PCA(n_components=2)
    pca_vector = pca.fit_transform(vector)

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit(pca_vector)

    cluster1, cluster2 = [], []

    for vec, cluster_number in zip(pca_vector, clusters.labels_):
        if cluster_number == 0:
            cluster1.append(vec)
        else:
            cluster2.append(vec)
    cluster1, cluster2 = np.array(cluster1), np.array(cluster2)

    plt.figure(figsize=(10, 5), dpi=150, tight_layout="True")
    plt.subplot(1, 2, 1)
    plt.title("K-Means Result")
    plt.scatter(cluster1[:, 0], cluster1[:, 1], label="cluster1")
    plt.scatter(cluster2[:, 0], cluster2[:, 1], label="cluster2")
    plt.legend(fontsize=14)
    plt.subplot(1, 2, 2)
    plt.scatter(pca_vector[1:30, 0], pca_vector[1:30, 1], label=categories[0])
    plt.scatter(pca_vector[31:60, 0], pca_vector[31:60, 1], label=categories[1])
    plt.legend(fontsize=14)
    plt.title("Groung Truth")
    plt.savefig("kmeans.png")


def main(args: argparse.Namespace) -> None:
    texts, labels = get_data(args.num_files, args.categories)
    morpeme_texts = wakati(texts)
    vector = tfidf_vectorizer(morpeme_texts)
    hierarchical_cluster(vector, labels)
    kmean_cluster(vector, args.categories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="an example program")
    parser.add_argument(
        "--num_files", default=30, type=int, help="number of text files"
    )
    parser.add_argument(
        "--categories",
        default=["smax", "sports-watch"],
        nargs="*",
        type=str,
        help="list of article categories",
    )
    args = parser.parse_args()
    main(args)
