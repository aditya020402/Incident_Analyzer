import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
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
# STEP 1: HDBSCAN Clustering
# ============================================================================

print("\n" + "="*80)
print("STEP 1: HDBSCAN CLUSTERING")
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
    min_cluster_size=15,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1,
    cluster_selection_epsilon=0.0
)

hdbscan_labels = hdbscan_model.fit_predict(embeddings_reduced)
n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_hdbscan_outliers = (hdbscan_labels == -1).sum()

print(f"HDBSCAN found {n_hdbscan_clusters} clusters")
print(f"HDBSCAN outliers: {n_hdbscan_outliers} ({n_hdbscan_outliers/len(hdbscan_labels)*100:.1f}%)")

# Initialize final labels
final_labels = hdbscan_labels.copy()
max_hdbscan_label = hdbscan_labels.max()

# ============================================================================
# STEP 2: KNN Voting with Median Intra-Cluster Distance Threshold
# ============================================================================

print("\n" + "="*80)
print("STEP 2: KNN VOTING (MEDIAN INTRA-CLUSTER DISTANCE)")
print("="*80)

outlier_mask_knn = final_labels == -1
n_outliers_for_knn = outlier_mask_knn.sum()

if n_outliers_for_knn > 0:
    print(f"Attempting KNN assignment for {n_outliers_for_knn} outliers...")
    
    outlier_indices = np.where(outlier_mask_knn)[0]
    outlier_embeddings = embeddings_reduced[outlier_mask_knn]
    
    # Get clustered points
    clustered_mask = final_labels != -1
    clustered_embeddings = embeddings_reduced[clustered_mask]
    clustered_labels = final_labels[clustered_mask]
    
    # Calculate median intra-cluster distance
    print("Calculating median intra-cluster distance...")
    intra_cluster_distances = []
    
    for cluster_id in set(clustered_labels):
        cluster_mask = final_labels == cluster_id
        cluster_points = embeddings_reduced[cluster_mask]
        
        if len(cluster_points) > 1:
            cluster_center = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - cluster_center, axis=1)
            intra_cluster_distances.extend(dists)
    
    # Use MEDIAN as threshold
    distance_threshold = np.median(intra_cluster_distances)
    print(f"Median intra-cluster distance (threshold): {distance_threshold:.4f}")
    
    # Calculate distances from outliers to all clustered points
    distances = cdist(outlier_embeddings, clustered_embeddings, metric='euclidean')
    
    assigned_count = 0
    k = 5  # Number of neighbors for voting
    
    for i, outlier_idx in enumerate(outlier_indices):
        # Find k nearest neighbors
        nearest_k_indices = np.argsort(distances[i])[:k]
        nearest_k_distances = distances[i][nearest_k_indices]
        
        # Only assign if closest neighbor is within median threshold
        if nearest_k_distances[0] <= distance_threshold:
            # Get labels of k nearest neighbors
            neighbor_labels = clustered_labels[nearest_k_indices]
            
            # Majority voting among k neighbors
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            assigned_cluster = unique[np.argmax(counts)]
            
            final_labels[outlier_idx] = assigned_cluster
            assigned_count += 1
    
    print(f"KNN assigned {assigned_count} outliers to existing clusters")
    print(f"Remaining outliers after KNN: {(final_labels == -1).sum()}")

# ============================================================================
# STEP 3: DBSCAN on Remaining Outliers
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DBSCAN ON REMAINING OUTLIERS")
print("="*80)

outlier_mask_dbscan = final_labels == -1
n_outliers_for_dbscan = outlier_mask_dbscan.sum()

if n_outliers_for_dbscan > 0:
    print(f"Clustering {n_outliers_for_dbscan} remaining outliers with DBSCAN...")
    
    outlier_embeddings = embeddings_reduced[outlier_mask_dbscan]
    outlier_indices = np.where(outlier_mask_dbscan)[0]
    
    # DBSCAN with moderate settings
    dbscan = DBSCAN(
        eps=1.5,
        min_samples=3,
        metric='euclidean',
        n_jobs=-1
    )
    
    dbscan_labels = dbscan.fit_predict(outlier_embeddings)
    
    # Count DBSCAN results
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_dbscan_outliers = (dbscan_labels == -1).sum()
    
    print(f"DBSCAN found {n_dbscan_clusters} new clusters")
    print(f"DBSCAN still marked {n_dbscan_outliers} as final outliers")
    
    # Offset DBSCAN labels to avoid collision with HDBSCAN clusters
    dbscan_labels_offset = dbscan_labels.copy()
    dbscan_labels_offset[dbscan_labels_offset != -1] += max_hdbscan_label + 1
    
    # Update final labels
    final_labels[outlier_indices] = dbscan_labels_offset
else:
    n_dbscan_clusters = 0
    n_dbscan_outliers = 0
    print("No outliers remaining after KNN voting")

n_final_outliers = (final_labels == -1).sum()

# ============================================================================
# STEP 4: Generate Topic Names for All Clusters
# ============================================================================

print("\n" + "="*80)
print("STEP 4: GENERATING TOPIC NAMES")
print("="*80)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

# Get all unique cluster IDs (excluding outliers)
unique_clusters = sorted([c for c in set(final_labels) if c != -1])

print(f"Generating topic names for {len(unique_clusters)} clusters...")

# Group documents by cluster
topic_docs = {}
for idx, cluster_id in enumerate(final_labels):
    if cluster_id != -1:
        if cluster_id not in topic_docs:
            topic_docs[cluster_id] = []
        topic_docs[cluster_id].append(descriptions[idx])

# Extract keywords for each cluster
words_per_topic = {}
for cluster_id in unique_clusters:
    docs = topic_docs.get(cluster_id, [])
    
    if len(docs) > 0:
        topic_text = ' '.join(docs)
        
        try:
            word_freq = vectorizer_model.fit_transform([topic_text])
            feature_names = vectorizer_model.get_feature_names_out()
            frequencies = word_freq.toarray()[0]
            top_indices = frequencies.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices if frequencies[i] > 0]
            words_per_topic[cluster_id] = top_words if len(top_words) > 0 else ['unknown']
        except:
            words_per_topic[cluster_id] = ['unknown']

print(f"Generated topic names for all {len(unique_clusters)} clusters")

# Create topic names
topic_names = {}
for cluster_id, words in words_per_topic.items():
    topic_names[cluster_id] = '_'.join(words[:3])

# Identify cluster sources
cluster_sources = {}
for cluster_id in unique_clusters:
    if cluster_id <= max_hdbscan_label:
        cluster_sources[cluster_id] = 'HDBSCAN'
    else:
        cluster_sources[cluster_id] = 'DBSCAN'

# Calculate probabilities
probabilities = np.zeros(len(final_labels))
for cluster_id in unique_clusters:
    mask = final_labels == cluster_id
    if mask.sum() > 0:
        cluster_center = embeddings_reduced[mask].mean(axis=0)
        distances_to_center = np.linalg.norm(embeddings_reduced[mask] - cluster_center, axis=1)
        max_dist = distances_to_center.max() if distances_to_center.max() > 0 else 1
        probabilities[mask] = 1 - (distances_to_center / max_dist * 0.5)

# ============================================================================
# STEP 5: Add Results to DataFrame
# ============================================================================

df['cluster_id'] = final_labels
df['cluster_probability'] = probabilities
df['cluster_name'] = df['cluster_id'].map(lambda x: topic_names.get(x, 'outlier'))
df['cluster_source'] = df['cluster_id'].map(lambda x: cluster_sources.get(x, 'outlier'))

# ============================================================================
# STEP 6: Generate Statistics
# ============================================================================

print("\n" + "="*80)
print("GENERATING CLUSTER STATISTICS")
print("="*80)

cluster_stats = []
for cluster_id in unique_clusters:
    cluster_docs = df[df['cluster_id'] == cluster_id]
    
    topic_keywords = ' | '.join(words_per_topic.get(cluster_id, ['N/A'])[:10])
    
    stats = {
        'cluster_id': cluster_id,
        'cluster_name': topic_names.get(cluster_id, 'unknown'),
        'cluster_source': cluster_sources.get(cluster_id, 'unknown'),
        'topic_words': topic_keywords,
        'count': len(cluster_docs),
        'avg_probability': cluster_docs['cluster_probability'].mean(),
        'avg_alert_count': cluster_docs['alert_count'].mean(),
        'unique_services': cluster_docs['bp_impacted_service'].nunique(),
        'unique_business_streams': cluster_docs['business_stream_name'].nunique(),
        'sample_descriptions': ' | '.join(cluster_docs['bp_description_cleaned'].head(2).tolist())
    }
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("\nSaving results...")

output_dir = Path('clusters')
output_dir.mkdir(exist_ok=True)

# Save summary
cluster_stats_df.to_csv(output_dir / 'cluster_summary.csv', index=False)

# Save complete data
df.to_csv(output_dir / 'all_incidents_clustered.csv', index=False)

# Save individual cluster files
print("\nCreating individual cluster files...")
for cluster_id in unique_clusters:
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    if len(cluster_df) > 0:
        topic_name = topic_names.get(cluster_id, f'cluster_{cluster_id}')
        source = cluster_sources.get(cluster_id, 'unknown')
        clean_name = topic_name.replace(' ', '_').replace('/', '_')[:50]
        filename = f'cluster_{cluster_id}_{source}_{clean_name}.csv'
        cluster_df.to_csv(output_dir / filename, index=False)
        print(f"  {filename}: {len(cluster_df)} incidents")

# Save outliers to outliers.csv
if n_final_outliers > 0:
    outliers_df = df[df['cluster_id'] == -1]
    outliers_df.to_csv(output_dir / 'outliers.csv', index=False)
    print(f"\nSaved {len(outliers_df)} final outliers to outliers.csv")

# ============================================================================
# STEP 8: Display Final Summary
# ============================================================================

print("\n" + "="*80)
print("FINAL CLUSTERING SUMMARY")
print("="*80)
print(f"Total incidents: {len(df)}")

print(f"\nProcessing Flow:")
print(f"  1. HDBSCAN: {n_hdbscan_clusters} clusters, {n_hdbscan_outliers} outliers")
print(f"  2. KNN Voting: Assigned {n_hdbscan_outliers - n_outliers_for_dbscan} outliers")
print(f"  3. DBSCAN: {n_dbscan_clusters} new clusters from {n_outliers_for_dbscan} remaining outliers")
print(f"  4. Final outliers: {n_final_outliers} ({n_final_outliers/len(df)*100:.2f}%)")

print(f"\nFinal Results:")
print(f"  Total clusters: {len(unique_clusters)}")
print(f"    - HDBSCAN clusters: {n_hdbscan_clusters}")
print(f"    - DBSCAN clusters: {n_dbscan_clusters}")
print(f"  Final outliers: {n_final_outliers}")

cluster_sizes = pd.Series(final_labels[final_labels != -1]).value_counts()
print(f"\nCluster size statistics:")
print(f"  Smallest cluster: {cluster_sizes.min()}")
print(f"  Largest cluster: {cluster_sizes.max()}")
print(f"  Average cluster size: {cluster_sizes.mean():.2f}")
print(f"  Median cluster size: {cluster_sizes.median():.0f}")

# Show breakdown by source
hdbscan_clusters_df = cluster_stats_df[cluster_stats_df['cluster_source'] == 'HDBSCAN']
dbscan_clusters_df = cluster_stats_df[cluster_stats_df['cluster_source'] == 'DBSCAN']

print(f"\nHDBSCAN clusters: {len(hdbscan_clusters_df)} clusters, {hdbscan_clusters_df['count'].sum()} incidents")
if len(dbscan_clusters_df) > 0:
    print(f"DBSCAN clusters: {len(dbscan_clusters_df)} clusters, {dbscan_clusters_df['count'].sum()} incidents")

print(f"\nTop 20 Largest Clusters:")
print(cluster_stats_df.nlargest(20, 'count')[
    ['cluster_id', 'cluster_source', 'cluster_name', 'count']
].to_string(index=False))

if n_dbscan_clusters > 0:
    print(f"\nDBSCAN-created clusters (from remaining outliers):")
    print(dbscan_clusters_df[['cluster_id', 'cluster_name', 'count']].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)
