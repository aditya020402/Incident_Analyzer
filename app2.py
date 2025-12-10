import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan
from bertopic import BERTopic
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# [Steps 1-3: Load, Clean, Prepare - Same as before]
# ============================================================================

print("Loading data...")
df_excel = pd.read_excel('incidents.xlsx')
df_parquet1 = pd.read_parquet('embeddings1.parquet')
df_parquet2 = pd.read_parquet('embeddings2.parquet')

df_excel['incident_id'] = df_excel['incident_id'].astype(str).str.strip()
df_parquet1['incident_id'] = df_parquet1['incident_id'].astype(str).str.strip()
df_parquet2['incident_id'] = df_parquet2['incident_id'].astype(str).str.strip()

embedding_dict = {str(row['incident_id']).strip(): row['embedding'] 
                  for _, row in df_parquet1.iterrows()}

for _, row in df_parquet2.iterrows():
    incident_id = str(row['incident_id']).strip()
    if incident_id not in embedding_dict:
        embedding_dict[incident_id] = row['embedding']

df_excel['embedding'] = df_excel['incident_id'].apply(
    lambda x: embedding_dict.get(str(x).strip(), None)
)

df = df_excel[df_excel['embedding'].notna()].copy().reset_index(drop=True)
print(f"Loaded {len(df)} records with embeddings")

print("Cleaning text...")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['bp_description_cleaned'] = df['bp_description'].apply(clean_text)
df = df[df['bp_description_cleaned'].str.len() > 0]

print("Removing duplicates...")
df = df.drop_duplicates(subset=['bp_description_cleaned'], keep='first').reset_index(drop=True)
print(f"Unique incidents: {len(df)}")

print("Preparing embeddings...")
descriptions = df['bp_description_cleaned'].tolist()

if isinstance(df['embedding'].iloc[0], str):
    embeddings = np.array([eval(x) for x in df['embedding']])
elif isinstance(df['embedding'].iloc[0], list):
    embeddings = np.array(df['embedding'].tolist())
else:
    embeddings = np.vstack(df['embedding'].values)

embeddings_normalized = normalize(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# ============================================================================
# STEP 4: Initial Clustering with HDBSCAN
# ============================================================================

print("\n" + "="*80)
print("STEP 1: INITIAL CLUSTERING WITH HDBSCAN")
print("="*80)

umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=25,
    min_dist=0.0,
    metric='cosine',
    low_memory=True,
    random_state=42
)

embeddings_reduced = umap_model.fit_transform(embeddings_normalized)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,         # Moderate size to avoid over-fragmentation
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom',  # eom is more stable
    prediction_data=True,
    core_dist_n_jobs=-1,
    cluster_selection_epsilon=0.0
)

initial_labels = hdbscan_model.fit_predict(embeddings_reduced)
n_initial_clusters = len(set(initial_labels)) - (1 if -1 in initial_labels else 0)
n_initial_outliers = (initial_labels == -1).sum()

print(f"Initial clusters: {n_initial_clusters}")
print(f"Initial outliers: {n_initial_outliers} ({n_initial_outliers/len(initial_labels)*100:.1f}%)")

# ============================================================================
# STEP 5: MERGE SIMILAR CLUSTERS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: MERGING SIMILAR CLUSTERS")
print("="*80)

# Calculate cluster centroids
cluster_ids = [c for c in set(initial_labels) if c != -1]
cluster_centroids = {}
cluster_sizes = {}

for cluster_id in cluster_ids:
    mask = initial_labels == cluster_id
    cluster_centroids[cluster_id] = embeddings_reduced[mask].mean(axis=0)
    cluster_sizes[cluster_id] = mask.sum()

print(f"Calculating similarity between {len(cluster_ids)} clusters...")

# Calculate pairwise distances between cluster centroids
centroid_matrix = np.array([cluster_centroids[cid] for cid in cluster_ids])
centroid_distances = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

# Define similarity threshold (tune this based on your data)
similarity_threshold = 0.8  # Clusters closer than this will be merged

# Use hierarchical clustering to merge similar clusters
from scipy.cluster.hierarchy import linkage, fcluster

# Perform hierarchical clustering on cluster centroids
linkage_matrix = linkage(centroid_matrix, method='average', metric='euclidean')

# Cut the dendrogram at similarity threshold
merged_cluster_labels = fcluster(linkage_matrix, t=similarity_threshold, criterion='distance')

# Create mapping from old cluster IDs to new merged cluster IDs
cluster_id_mapping = {}
for idx, old_cluster_id in enumerate(cluster_ids):
    new_cluster_id = merged_cluster_labels[idx] - 1  # Convert to 0-indexed
    cluster_id_mapping[old_cluster_id] = new_cluster_id

# Apply mapping to all labels
merged_labels = initial_labels.copy()
for old_id, new_id in cluster_id_mapping.items():
    merged_labels[merged_labels == old_id] = new_id

n_merged_clusters = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
print(f"After merging: {n_merged_clusters} clusters (reduced from {n_initial_clusters})")
print(f"Merged {n_initial_clusters - n_merged_clusters} duplicate clusters")

# ============================================================================
# STEP 6: STRICT KNN VOTING FOR VERY CLOSE OUTLIERS ONLY
# ============================================================================

print("\n" + "="*80)
print("STEP 3: KNN ASSIGNMENT FOR VERY CLOSE OUTLIERS ONLY")
print("="*80)

final_labels = merged_labels.copy()
outlier_mask = final_labels == -1
n_outliers_before = outlier_mask.sum()

if n_outliers_before > 0:
    outlier_indices = np.where(outlier_mask)[0]
    outlier_embeddings = embeddings_reduced[outlier_mask]
    
    # Get clustered points
    clustered_mask = final_labels != -1
    clustered_embeddings = embeddings_reduced[clustered_mask]
    clustered_labels = final_labels[clustered_mask]
    
    # Calculate distances from each outlier to all clustered points
    distances = cdist(outlier_embeddings, clustered_embeddings, metric='euclidean')
    
    # Define STRICT distance threshold - only assign if very close
    # Use 10th percentile of intra-cluster distances as threshold
    intra_cluster_distances = []
    for cluster_id in set(clustered_labels):
        cluster_points = embeddings_reduced[final_labels == cluster_id]
        if len(cluster_points) > 1:
            cluster_center = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - cluster_center, axis=1)
            intra_cluster_distances.extend(dists)
    
    distance_threshold = np.percentile(intra_cluster_distances, 50)  # 50th percentile = median
    print(f"Distance threshold for KNN assignment: {distance_threshold:.4f}")
    
    assigned_count = 0
    
    for i, outlier_idx in enumerate(outlier_indices):
        # Find k nearest neighbors
        k = 5
        nearest_k_indices = np.argsort(distances[i])[:k]
        nearest_k_distances = distances[i][nearest_k_indices]
        
        # Only assign if closest neighbor is within threshold
        if nearest_k_distances[0] <= distance_threshold:
            # Get labels of k nearest neighbors
            neighbor_labels = clustered_labels[nearest_k_indices]
            
            # Assign to most common cluster
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            assigned_cluster = unique[np.argmax(counts)]
            
            final_labels[outlier_idx] = assigned_cluster
            assigned_count += 1
    
    print(f"Assigned {assigned_count} close outliers to nearest clusters")
    print(f"Remaining outliers: {(final_labels == -1).sum()}")

# ============================================================================
# STEP 7: DBSCAN FOR REMAINING OUTLIERS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: DBSCAN FOR REMAINING OUTLIERS")
print("="*80)

remaining_outliers = (final_labels == -1).sum()

if remaining_outliers > 0:
    print(f"Clustering {remaining_outliers} remaining outliers with DBSCAN...")
    
    outlier_mask = final_labels == -1
    outlier_embeddings_remain = embeddings_reduced[outlier_mask]
    
    # Use DBSCAN with moderate settings
    dbscan = DBSCAN(
        eps=1.2,           # Distance threshold
        min_samples=3,     # Minimum 3 points to form cluster
        metric='euclidean',
        n_jobs=-1
    )
    
    dbscan_labels = dbscan.fit_predict(outlier_embeddings_remain)
    
    # Count how many were clustered
    n_dbscan_outliers = (dbscan_labels == -1).sum()
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    print(f"DBSCAN found {n_dbscan_clusters} new clusters")
    print(f"DBSCAN still marked {n_dbscan_outliers} as outliers")
    
    # Offset labels to avoid collision with existing clusters
    max_label = final_labels[final_labels != -1].max()
    dbscan_labels_offset = dbscan_labels.copy()
    dbscan_labels_offset[dbscan_labels_offset != -1] += max_label + 1
    
    # Update final labels
    final_labels[outlier_mask] = dbscan_labels_offset

# ============================================================================
# STEP 8: Calculate Final Statistics
# ============================================================================

n_final_outliers = (final_labels == -1).sum()
n_total_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)

print("\n" + "="*80)
print("FINAL CLUSTERING RESULTS")
print("="*80)
print(f"Total incidents: {len(final_labels)}")
print(f"Total clusters: {n_total_clusters}")
print(f"Final outliers: {n_final_outliers} ({n_final_outliers/len(final_labels)*100:.2f}%)")

cluster_sizes = pd.Series(final_labels[final_labels != -1]).value_counts()
print(f"\nCluster size statistics:")
print(f"  Smallest cluster: {cluster_sizes.min()}")
print(f"  Largest cluster: {cluster_sizes.max()}")
print(f"  Average cluster size: {cluster_sizes.mean():.2f}")
print(f"  Median cluster size: {cluster_sizes.median():.0f}")

# ============================================================================
# STEP 9: Create Topic Labels with BERTopic
# ============================================================================

print("\nCreating topic labels...")

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

topic_docs = {}
for idx, topic in enumerate(final_labels):
    if topic != -1:
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(descriptions[idx])

words_per_topic = {}
for topic_id, docs in topic_docs.items():
    if len(docs) > 0:
        topic_text = ' '.join(docs)
        try:
            word_freq = vectorizer_model.fit_transform([topic_text])
            feature_names = vectorizer_model.get_feature_names_out()
            frequencies = word_freq.toarray()[0]
            top_indices = frequencies.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices if frequencies[i] > 0]
            words_per_topic[topic_id] = top_words if len(top_words) > 0 else ['unknown']
        except:
            words_per_topic[topic_id] = ['unknown']

# Calculate probabilities
probabilities = np.zeros(len(final_labels))
for topic_id in set(final_labels):
    if topic_id != -1:
        mask = final_labels == topic_id
        if mask.sum() > 0:
            cluster_center = embeddings_reduced[mask].mean(axis=0)
            distances = np.linalg.norm(embeddings_reduced[mask] - cluster_center, axis=1)
            max_dist = distances.max() if distances.max() > 0 else 1
            probabilities[mask] = 1 - (distances / max_dist * 0.5)

# ============================================================================
# STEP 10: Save Results
# ============================================================================

df['cluster_id'] = final_labels
df['cluster_probability'] = probabilities

topic_names = {}
for topic_id, words in words_per_topic.items():
    topic_names[topic_id] = '_'.join(words[:3])

df['cluster_name'] = df['cluster_id'].map(lambda x: topic_names.get(x, 'outlier'))

print("\nSaving results...")

output_dir = Path('clusters')
output_dir.mkdir(exist_ok=True)

unique_topics = sorted([t for t in df['cluster_id'].unique() if t != -1])

cluster_stats = []
for cluster_id in unique_topics:
    cluster_docs = df[df['cluster_id'] == cluster_id]
    
    topic_keywords = ' | '.join(words_per_topic.get(cluster_id, ['N/A'])[:10])
    
    stats = {
        'cluster_id': cluster_id,
        'cluster_name': topic_names.get(cluster_id, 'unknown'),
        'topic_words': topic_keywords,
        'count': len(cluster_docs),
        'avg_probability': cluster_docs['cluster_probability'].mean(),
        'avg_alert_count': cluster_docs['alert_count'].mean(),
        'unique_services': cluster_docs['bp_impacted_service'].nunique(),
        'unique_business_streams': cluster_docs['business_stream_name'].nunique(),
        'sample_descriptions': ' | '.join(cluster_docs['bp_description_cleaned'].head(3).tolist())
    }
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)
cluster_stats_df.to_csv(output_dir / 'cluster_summary.csv', index=False)
df.to_csv(output_dir / 'all_incidents_clustered.csv', index=False)

# Save individual cluster files
print("\nCreating individual cluster files...")
for cluster_id in unique_topics:
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    if len(cluster_df) > 0:
        topic_name = topic_names.get(cluster_id, f'topic_{cluster_id}')
        clean_name = topic_name.replace(' ', '_').replace('/', '_')[:50]
        filename = f'cluster_{cluster_id}_{clean_name}.csv'
        cluster_df.to_csv(output_dir / filename, index=False)

if n_final_outliers > 0:
    outliers_df = df[df['cluster_id'] == -1]
    outliers_df.to_csv(output_dir / 'cluster_outliers.csv', index=False)

# ============================================================================
# STEP 11: Display Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total incidents: {len(df)}")
print(f"Total clusters: {n_total_clusters}")
print(f"Final outliers: {n_final_outliers} ({n_final_outliers/len(df)*100:.2f}%)")

print(f"\nTop 20 Clusters:")
print(cluster_stats_df.nlargest(20, 'count')[
    ['cluster_id', 'cluster_name', 'count']
].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)
