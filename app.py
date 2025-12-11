import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from bertopic import BERTopic
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Data from Excel and Parquet
# ============================================================================

print("Loading data from Excel and Parquet files...")

# Load Excel file
df_excel = pd.read_excel('incidents.xlsx')
print(f"Loaded {len(df_excel)} records from Excel")

# Load Parquet file with embeddings
df_parquet = pd.read_parquet('embeddings.parquet')
print(f"Loaded {len(df_parquet)} embeddings from Parquet")

# Ensure incident_id is string and stripped
df_excel['incident_id'] = df_excel['incident_id'].astype(str).str.strip()
df_parquet['incident_id'] = df_parquet['incident_id'].astype(str).str.strip()

# Create embedding dictionary
embedding_dict = {str(row['incident_id']).strip(): row['embedding'] 
                  for _, row in df_parquet.iterrows()}

# Add embeddings to Excel data
df_excel['embedding'] = df_excel['incident_id'].apply(
    lambda x: embedding_dict.get(str(x).strip(), None)
)

# Keep only records with embeddings
df = df_excel[df_excel['embedding'].notna()].copy()
print(f"After filtering: {len(df)} records with embeddings")

# ============================================================================
# STEP 2: Clean bp_description Text
# ============================================================================

print("\nCleaning bp_description field...")

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

df['bp_description_cleaned'] = df['bp_description'].apply(clean_text)

# Remove empty descriptions
df_before_empty = len(df)
df = df[df['bp_description_cleaned'].str.len() > 0]
print(f"Removed {df_before_empty - len(df)} records with empty descriptions")

# ============================================================================
# STEP 3: Remove Duplicates
# ============================================================================

print("\nRemoving duplicate descriptions...")

duplicate_count = df.duplicated(subset=['bp_description_cleaned'], keep='first').sum()
print(f"Found {duplicate_count} duplicate descriptions")

df_before_dedup = len(df)
df = df.drop_duplicates(subset=['bp_description_cleaned'], keep='first')
print(f"Removed {df_before_dedup - len(df)} duplicate incidents")
print(f"Remaining records: {len(df)}")

df = df.reset_index(drop=True)

# ============================================================================
# STEP 4: Prepare Embeddings
# ============================================================================

print("\nPreparing embeddings...")

descriptions = df['bp_description_cleaned'].tolist()

# Parse embeddings
if isinstance(df['embedding'].iloc[0], str):
    embeddings = np.array([eval(x) for x in df['embedding']])
elif isinstance(df['embedding'].iloc[0], list):
    embeddings = np.array(df['embedding'].tolist())
else:
    embeddings = np.vstack(df['embedding'].values)

embeddings_normalized = normalize(embeddings)

print(f"Processing {len(descriptions)} unique incidents")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# ============================================================================
# STEP 5: BERTopic Clustering
# ============================================================================

print("\n" + "="*80)
print("STEP 5: BERTOPIC CLUSTERING")
print("="*80)

# Configure UMAP
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=25,
    min_dist=0.0,
    metric='cosine',
    low_memory=True,
    random_state=42
)

# Configure HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1,
    cluster_selection_epsilon=0.0
)

# Initialize BERTopic
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
    nr_topics='auto',
    calculate_probabilities=False
)

# Fit BERTopic
print("Fitting BERTopic model...")
topics, probabilities = topic_model.fit_transform(descriptions, embeddings_normalized)

# Get reduced embeddings for KNN and DBSCAN
embeddings_reduced = umap_model.fit_transform(embeddings_normalized)

n_bertopic_clusters = len(set(topics)) - (1 if -1 in topics else 0)
n_bertopic_outliers = (topics == -1).sum()

print(f"BERTopic found {n_bertopic_clusters} topics")
print(f"BERTopic outliers: {n_bertopic_outliers} ({n_bertopic_outliers/len(topics)*100:.1f}%)")

# Initialize final labels
final_labels = np.array(topics).copy()
max_bertopic_label = max([t for t in topics if t != -1]) if n_bertopic_clusters > 0 else -1

# ============================================================================
# STEP 6: KNN Voting with Median Intra-Cluster Distance
# ============================================================================

print("\n" + "="*80)
print("STEP 6: KNN VOTING (MEDIAN INTRA-CLUSTER DISTANCE)")
print("="*80)

outlier_mask_knn = final_labels == -1
n_outliers_for_knn = outlier_mask_knn.sum()

if n_outliers_for_knn > 0 and n_bertopic_clusters > 0:
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
    
    if len(intra_cluster_distances) > 0:
        distance_threshold = np.median(intra_cluster_distances)
        print(f"Median intra-cluster distance (threshold): {distance_threshold:.4f}")
        
        # Calculate distances
        distances = cdist(outlier_embeddings, clustered_embeddings, metric='euclidean')
        
        assigned_count = 0
        k = 5
        
        for i, outlier_idx in enumerate(outlier_indices):
            nearest_k_indices = np.argsort(distances[i])[:k]
            nearest_k_distances = distances[i][nearest_k_indices]
            
            # Check if closest neighbor is within median threshold
            if nearest_k_distances[0] <= distance_threshold:
                neighbor_labels = clustered_labels[nearest_k_indices]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                assigned_cluster = unique[np.argmax(counts)]
                final_labels[outlier_idx] = assigned_cluster
                assigned_count += 1
        
        print(f"KNN assigned {assigned_count} outliers to existing topics")
        print(f"Remaining outliers after KNN: {(final_labels == -1).sum()}")
    else:
        print("No intra-cluster distances to calculate threshold")
else:
    print("No outliers to process with KNN or no clusters exist")

# ============================================================================
# STEP 7: DBSCAN on Remaining Outliers
# ============================================================================

print("\n" + "="*80)
print("STEP 7: DBSCAN ON REMAINING OUTLIERS")
print("="*80)

outlier_mask_dbscan = final_labels == -1
n_outliers_for_dbscan = outlier_mask_dbscan.sum()

if n_outliers_for_dbscan > 0:
    print(f"Clustering {n_outliers_for_dbscan} remaining outliers with DBSCAN...")
    
    outlier_embeddings = embeddings_reduced[outlier_mask_dbscan]
    outlier_indices = np.where(outlier_mask_dbscan)[0]
    
    dbscan = DBSCAN(
        eps=1.5,
        min_samples=3,
        metric='euclidean',
        n_jobs=-1
    )
    
    dbscan_labels = dbscan.fit_predict(outlier_embeddings)
    
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_dbscan_outliers = (dbscan_labels == -1).sum()
    
    print(f"DBSCAN found {n_dbscan_clusters} new clusters")
    print(f"DBSCAN still marked {n_dbscan_outliers} as final outliers")
    
    # Offset DBSCAN labels to avoid collision
    dbscan_labels_offset = dbscan_labels.copy()
    dbscan_labels_offset[dbscan_labels_offset != -1] += max_bertopic_label + 1
    
    final_labels[outlier_indices] = dbscan_labels_offset
else:
    n_dbscan_clusters = 0
    n_dbscan_outliers = 0
    print("No outliers remaining after KNN voting")

n_final_outliers = (final_labels == -1).sum()

# ============================================================================
# STEP 8: Generate Topic Names for All Clusters
# ============================================================================

print("\n" + "="*80)
print("STEP 8: GENERATING TOPIC NAMES")
print("="*80)

# Get BERTopic topic info
topic_info = topic_model.get_topic_info()

# Create topic names dictionary for BERTopic clusters
bertopic_topic_names = {}
bertopic_topic_words = {}

for _, row in topic_info.iterrows():
    topic_id = row['Topic']
    if topic_id != -1:
        # Get topic words from BERTopic
        topic_words_list = topic_model.get_topic(topic_id)
        if topic_words_list:
            words = [word for word, _ in topic_words_list[:10]]
            bertopic_topic_words[topic_id] = words
            bertopic_topic_names[topic_id] = '_'.join(words[:3])
        else:
            bertopic_topic_words[topic_id] = ['unknown']
            bertopic_topic_names[topic_id] = 'unknown'

# Generate topic names for DBSCAN clusters
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

dbscan_topic_names = {}
dbscan_topic_words = {}

if n_dbscan_clusters > 0:
    print(f"Generating topic names for {n_dbscan_clusters} DBSCAN clusters...")
    
    dbscan_cluster_ids = [c for c in set(final_labels) if c > max_bertopic_label]
    
    for cluster_id in dbscan_cluster_ids:
        cluster_mask = final_labels == cluster_id
        cluster_docs = [descriptions[i] for i in np.where(cluster_mask)[0]]
        
        if len(cluster_docs) > 0:
            topic_text = ' '.join(cluster_docs)
            
            try:
                word_freq = vectorizer_model.fit_transform([topic_text])
                feature_names = vectorizer_model.get_feature_names_out()
                frequencies = word_freq.toarray()[0]
                top_indices = frequencies.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices if frequencies[i] > 0]
                
                dbscan_topic_words[cluster_id] = top_words if len(top_words) > 0 else ['unknown']
                dbscan_topic_names[cluster_id] = '_'.join(top_words[:3]) if len(top_words) > 0 else 'unknown'
            except:
                dbscan_topic_words[cluster_id] = ['unknown']
                dbscan_topic_names[cluster_id] = 'unknown'

# Combine topic names
all_topic_names = {**bertopic_topic_names, **dbscan_topic_names}
all_topic_words = {**bertopic_topic_words, **dbscan_topic_words}

# Identify cluster sources
cluster_sources = {}
for cluster_id in set(final_labels):
    if cluster_id == -1:
        cluster_sources[cluster_id] = 'outlier'
    elif cluster_id <= max_bertopic_label:
        cluster_sources[cluster_id] = 'BERTopic'
    else:
        cluster_sources[cluster_id] = 'DBSCAN'

print(f"Generated topic names for all {len(all_topic_names)} clusters")

# Calculate probabilities
probabilities_final = np.zeros(len(final_labels))
for cluster_id in set(final_labels):
    if cluster_id != -1:
        mask = final_labels == cluster_id
        if mask.sum() > 0:
            cluster_center = embeddings_reduced[mask].mean(axis=0)
            distances_to_center = np.linalg.norm(embeddings_reduced[mask] - cluster_center, axis=1)
            max_dist = distances_to_center.max() if distances_to_center.max() > 0 else 1
            probabilities_final[mask] = 1 - (distances_to_center / max_dist * 0.5)

# ============================================================================
# STEP 9: Add Results to DataFrame
# ============================================================================

df['cluster_id'] = final_labels
df['cluster_probability'] = probabilities_final
df['cluster_name'] = df['cluster_id'].map(lambda x: all_topic_names.get(x, 'outlier'))
df['cluster_source'] = df['cluster_id'].map(lambda x: cluster_sources.get(x, 'outlier'))

# ============================================================================
# STEP 10: Generate Statistics
# ============================================================================

print("\nGenerating cluster statistics...")

unique_clusters = sorted([c for c in set(final_labels) if c != -1])

cluster_stats = []
for cluster_id in unique_clusters:
    cluster_docs = df[df['cluster_id'] == cluster_id]
    
    topic_keywords = ' | '.join(all_topic_words.get(cluster_id, ['N/A'])[:10])
    
    stats = {
        'cluster_id': cluster_id,
        'cluster_name': all_topic_names.get(cluster_id, 'unknown'),
        'cluster_source': cluster_sources.get(cluster_id, 'unknown'),
        'topic_words': topic_keywords,
        'count': len(cluster_docs),
        'avg_probability': cluster_docs['cluster_probability'].mean(),
        'avg_alert_count': cluster_docs['alert_count'].mean(),
        'unique_services': cluster_docs['bp_impacted_service'].nunique(),
        'unique_business_streams': cluster_docs['business_stream_name'].nunique(),
        'sample_description': cluster_docs['bp_description_cleaned'].iloc[0][:100] if len(cluster_docs) > 0 else ''
    }
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)

# ============================================================================
# STEP 11: Save Results
# ============================================================================

print("\nSaving results...")

output_dir = Path('clusters')
output_dir.mkdir(exist_ok=True)

# Save cluster summary
cluster_stats_df.to_csv(output_dir / 'cluster_summary.csv', index=False)

# Save complete results
df.to_csv(output_dir / 'all_incidents_clustered.csv', index=False)

# Save BERTopic model
topic_model.save(str(output_dir / 'bertopic_model'))

# Save individual cluster files
print("\nCreating individual cluster files...")
for cluster_id in unique_clusters:
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    if len(cluster_df) > 0:
        topic_name = all_topic_names.get(cluster_id, f'cluster_{cluster_id}')
        source = cluster_sources.get(cluster_id, 'unknown')
        clean_name = topic_name.replace(' ', '_').replace('/', '_')[:50]
        filename = f'cluster_{cluster_id}_{source}_{clean_name}.csv'
        cluster_df.to_csv(output_dir / filename, index=False)

# Save outliers
if n_final_outliers > 0:
    outliers_df = df[df['cluster_id'] == -1]
    outliers_df.to_csv(output_dir / 'outliers.csv', index=False)
    print(f"\nSaved {len(outliers_df)} final outliers to outliers.csv")

# ============================================================================
# STEP 12: Display Summary
# ============================================================================

print("\n" + "="*80)
print("CLUSTERING COMPLETE")
print("="*80)
print(f"Total incidents: {len(df)}")
print(f"Duplicates removed: {duplicate_count}")

print(f"\nProcessing Flow:")
print(f"  1. BERTopic: {n_bertopic_clusters} topics, {n_bertopic_outliers} outliers")
print(f"  2. KNN Voting: Assigned {n_bertopic_outliers - n_outliers_for_dbscan} outliers")
print(f"  3. DBSCAN: {n_dbscan_clusters} new clusters from {n_outliers_for_dbscan} remaining outliers")
print(f"  4. Final outliers: {n_final_outliers} ({n_final_outliers/len(df)*100:.2f}%)")

print(f"\nFinal Results:")
print(f"  Total clusters: {len(unique_clusters)}")
print(f"    - BERTopic clusters: {n_bertopic_clusters}")
print(f"    - DBSCAN clusters: {n_dbscan_clusters}")

cluster_sizes = pd.Series(final_labels[final_labels != -1]).value_counts()
if len(cluster_sizes) > 0:
    print(f"\nCluster size statistics:")
    print(f"  Smallest: {cluster_sizes.min()}")
    print(f"  Largest: {cluster_sizes.max()}")
    print(f"  Average: {cluster_sizes.mean():.2f}")
    print(f"  Median: {cluster_sizes.median():.0f}")

# Show breakdown by source
bertopic_clusters_df = cluster_stats_df[cluster_stats_df['cluster_source'] == 'BERTopic']
dbscan_clusters_df = cluster_stats_df[cluster_stats_df['cluster_source'] == 'DBSCAN']

print(f"\nBERTopic clusters: {len(bertopic_clusters_df)} clusters, {bertopic_clusters_df['count'].sum()} incidents")
if len(dbscan_clusters_df) > 0:
    print(f"DBSCAN clusters: {len(dbscan_clusters_df)} clusters, {dbscan_clusters_df['count'].sum()} incidents")

print(f"\nTop 20 Largest Clusters:")
print(cluster_stats_df.nlargest(20, 'count')[
    ['cluster_id', 'cluster_source', 'cluster_name', 'count', 'avg_probability']
].to_string(index=False))

if n_dbscan_clusters > 0:
    print(f"\nDBSCAN-created clusters (from remaining outliers):")
    print(dbscan_clusters_df[['cluster_id', 'cluster_name', 'count', 'topic_words']].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)
