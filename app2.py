import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan
from bertopic import BERTopic
from sklearn.cluster import DBSCAN, AgglomerativeClustering
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
# STEP 4: MULTI-STAGE CLUSTERING WITH min_cluster_size=3
# ============================================================================

print("\n" + "="*80)
print("AGGRESSIVE MULTI-STAGE CLUSTERING (min_cluster_size=3)")
print("="*80)

# Stage 1: UMAP with good preservation
print("\nStage 1: UMAP dimensionality reduction...")
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=25,
    min_dist=0.0,
    metric='cosine',
    low_memory=True,
    random_state=42
)

embeddings_reduced = umap_model.fit_transform(embeddings_normalized)
print(f"Reduced to {embeddings_reduced.shape[1]} dimensions")

# Stage 2: Very permissive HDBSCAN for main clusters
print("\nStage 2: HDBSCAN with min_cluster_size=3...")
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=3,          # Allow clusters as small as 3
    min_samples=1,               # Very permissive
    metric='euclidean',
    cluster_selection_method='leaf',  # Leaf finds more clusters
    prediction_data=True,
    core_dist_n_jobs=-1,
    cluster_selection_epsilon=0.0,
    alpha=1.0  # Default, controls cluster shape
)

hdbscan_labels = hdbscan_model.fit_predict(embeddings_reduced)
n_outliers_initial = (hdbscan_labels == -1).sum()
n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

print(f"HDBSCAN found {n_hdbscan_clusters} clusters")
print(f"Initial outliers: {n_outliers_initial} ({n_outliers_initial/len(hdbscan_labels)*100:.1f}%)")

# Stage 3: Assign outliers to nearest cluster using approximate approach
final_labels = hdbscan_labels.copy()

if n_outliers_initial > 0:
    print(f"\nStage 3: Assigning {n_outliers_initial} outliers to nearest clusters...")
    
    outlier_mask = final_labels == -1
    outlier_indices = np.where(outlier_mask)[0]
    
    # Get cluster embeddings
    clustered_mask = final_labels != -1
    clustered_labels = final_labels[clustered_mask]
    clustered_embeddings = embeddings_reduced[clustered_mask]
    
    # For each outlier, find nearest clustered point
    outlier_embeddings = embeddings_reduced[outlier_mask]
    
    # Use nearest neighbor approach
    from scipy.spatial.distance import cdist
    
    # Calculate distances in batches to save memory
    batch_size = 1000
    assigned = 0
    
    for i in range(0, len(outlier_indices), batch_size):
        batch_indices = outlier_indices[i:i+batch_size]
        batch_embeddings = embeddings_reduced[batch_indices]
        
        # Find k nearest clustered neighbors
        distances = cdist(batch_embeddings, clustered_embeddings, metric='euclidean')
        
        # For each outlier, find nearest 5 neighbors
        k = min(5, len(clustered_embeddings))
        nearest_k_indices = np.argsort(distances, axis=1)[:, :k]
        
        # Vote on cluster assignment
        for j, outlier_idx in enumerate(batch_indices):
            neighbor_labels = clustered_labels[nearest_k_indices[j]]
            # Assign to most common cluster among nearest neighbors
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            assigned_cluster = unique[np.argmax(counts)]
            final_labels[outlier_idx] = assigned_cluster
            assigned += 1
    
    print(f"Assigned {assigned} outliers to nearest clusters")

# Stage 4: Cluster remaining outliers with DBSCAN (if any)
remaining_outliers = (final_labels == -1).sum()

if remaining_outliers > 10:
    print(f"\nStage 4: Clustering {remaining_outliers} remaining outliers with DBSCAN...")
    
    outlier_mask = final_labels == -1
    outlier_embeddings_remain = embeddings_reduced[outlier_mask]
    
    # Use DBSCAN with very permissive settings
    dbscan = DBSCAN(eps=1.5, min_samples=2, metric='euclidean', n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(outlier_embeddings_remain)
    
    # Offset labels to avoid collision
    max_label = final_labels.max()
    dbscan_labels[dbscan_labels != -1] += max_label + 1
    
    # Update final labels
    final_labels[outlier_mask] = dbscan_labels
    
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {n_dbscan_clusters} additional clusters")

# Stage 5: Final outlier assignment with Agglomerative Clustering
remaining_outliers = (final_labels == -1).sum()

if remaining_outliers > 3:
    print(f"\nStage 5: Final pass - clustering {remaining_outliers} outliers with Agglomerative...")
    
    outlier_mask = final_labels == -1
    outlier_embeddings_final = embeddings_reduced[outlier_mask]
    
    # Determine number of clusters for remaining outliers
    n_final_clusters = max(1, remaining_outliers // 3)  # At least 3 per cluster
    n_final_clusters = min(n_final_clusters, remaining_outliers - 1)
    
    if n_final_clusters > 0 and remaining_outliers > 1:
        agg_clustering = AgglomerativeClustering(
            n_clusters=n_final_clusters,
            metric='euclidean',
            linkage='ward'
        )
        agg_labels = agg_clustering.fit_predict(outlier_embeddings_final)
        
        # Offset labels
        max_label = final_labels[final_labels != -1].max()
        agg_labels += max_label + 1
        
        final_labels[outlier_mask] = agg_labels
        print(f"Created {n_final_clusters} final clusters")
    elif remaining_outliers == 1:
        # Single outlier - assign to nearest cluster
        single_outlier_idx = np.where(outlier_mask)[0][0]
        single_outlier_emb = embeddings_reduced[single_outlier_idx].reshape(1, -1)
        
        clustered_mask = final_labels != -1
        clustered_embeddings = embeddings_reduced[clustered_mask]
        clustered_labels = final_labels[clustered_mask]
        
        distances = cdist(single_outlier_emb, clustered_embeddings, metric='euclidean')
        nearest_idx = np.argmin(distances)
        final_labels[single_outlier_idx] = clustered_labels[nearest_idx]

# ============================================================================
# STEP 5: Final Statistics
# ============================================================================

n_final_outliers = (final_labels == -1).sum()
n_total_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)

print("\n" + "="*80)
print("CLUSTERING RESULTS")
print("="*80)
print(f"Total incidents: {len(final_labels)}")
print(f"Total clusters: {n_total_clusters}")
print(f"Final outliers: {n_final_outliers} ({n_final_outliers/len(final_labels)*100:.2f}%)")

# Check cluster sizes
cluster_sizes = pd.Series(final_labels[final_labels != -1]).value_counts()
print(f"\nCluster size statistics:")
print(f"  Smallest cluster: {cluster_sizes.min()}")
print(f"  Largest cluster: {cluster_sizes.max()}")
print(f"  Average cluster size: {cluster_sizes.mean():.2f}")
print(f"  Median cluster size: {cluster_sizes.median():.0f}")
print(f"  Clusters with 3 members: {(cluster_sizes == 3).sum()}")
print(f"  Clusters with 4-10 members: {((cluster_sizes >= 4) & (cluster_sizes <= 10)).sum()}")
print(f"  Clusters with >10 members: {(cluster_sizes > 10).sum()}")

# ============================================================================
# STEP 6: Create BERTopic Model with Results
# ============================================================================

print("\nCreating BERTopic topic representations...")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# Create topic representations manually
vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

# Group documents by topic
topic_docs = {}
for idx, topic in enumerate(final_labels):
    if topic != -1:
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(descriptions[idx])

# Extract keywords for each topic
words_per_topic = {}
for topic_id, docs in topic_docs.items():
    if len(docs) > 0:
        # Combine all documents in topic
        topic_text = ' '.join(docs)
        
        # Get word frequencies
        try:
            word_freq = vectorizer_model.fit_transform([topic_text])
            feature_names = vectorizer_model.get_feature_names_out()
            
            # Get top words
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
            probabilities[mask] = 1 - (distances / max_dist * 0.5)  # Scale to 0.5-1.0

# ============================================================================
# STEP 7: Save Results
# ============================================================================

df['cluster_id'] = final_labels
df['cluster_probability'] = probabilities

# Create topic names
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
        'sample_descriptions': ' | '.join(cluster_docs['bp_description_cleaned'].head(2).tolist())
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

# Save any remaining outliers
if n_final_outliers > 0:
    outliers_df = df[df['cluster_id'] == -1]
    outliers_df.to_csv(output_dir / 'cluster_outliers.csv', index=False)
    print(f"\nRemaining outliers saved: {len(outliers_df)}")

# ============================================================================
# STEP 8: Final Summary
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total incidents: {len(df)}")
print(f"Total clusters: {n_total_clusters}")
print(f"Final outliers: {n_final_outliers} ({n_final_outliers/len(df)*100:.2f}%)")
print(f"Reduction from 24k: {24000 - n_final_outliers} incidents recovered")

print(f"\nSmall clusters (3-10 members): {((cluster_sizes >= 3) & (cluster_sizes <= 10)).sum()}")
print(f"Medium clusters (11-50 members): {((cluster_sizes >= 11) & (cluster_sizes <= 50)).sum()}")
print(f"Large clusters (>50 members): {(cluster_sizes > 50).sum()}")

print(f"\nTop 20 Largest Clusters:")
print(cluster_stats_df.nlargest(20, 'count')[
    ['cluster_id', 'cluster_name', 'count']
].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)
