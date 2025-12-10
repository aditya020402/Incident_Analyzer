from sklearn.metrics import silhouette_score

for n_dim in [5, 10, 15, 20, 30, 50]:
    umap_model = umap.UMAP(n_components=n_dim, metric='cosine', random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(descriptions, embeddings_normalized)
    
    # Evaluate
    reduced_emb = umap_model.fit_transform(embeddings_normalized)
    valid_topics = [t for t in topics if t != -1]
    if len(set(valid_topics)) > 1:
        score = silhouette_score(reduced_emb[topics != -1], 
                                [t for t in topics if t != -1])
        print(f"Dimensions: {n_dim}, Silhouette: {score:.4f}, Topics: {len(set(topics))-1}")
