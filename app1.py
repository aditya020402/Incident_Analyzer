import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan
from bertopic import BERTopic
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load and Join Data from Multiple Parquet Files
# ============================================================================

print("Loading data...")
df_excel = pd.read_excel('incidents.xlsx')
df_parquet1 = pd.read_parquet('embeddings1.parquet')
df_parquet2 = pd.read_parquet('embeddings2.parquet')

# Convert incident_id to string in all dataframes
df_excel['incident_id'] = df_excel['incident_id'].astype(str).str.strip()
df_parquet1['incident_id'] = df_parquet1['incident_id'].astype(str).str.strip()
df_parquet2['incident_id'] = df_parquet2['incident_id'].astype(str).str.strip()

# Create embedding dictionary from first parquet file
embedding_dict = {str(row['incident_id']).strip(): row['embedding'] 
                  for _, row in df_parquet1.iterrows()}

# Add embeddings from second parquet file (only if not already present)
for _, row in df_parquet2.iterrows():
    incident_id = str(row['incident_id']).strip()
    if incident_id not in embedding_dict:
        embedding_dict[incident_id] = row['embedding']

print(f"Total embeddings loaded: {len(embedding_dict)}")

# Add embeddings to Excel data
df_excel['embedding'] = df_excel['incident_id'].apply(
    lambda x: embedding_dict.get(str(x).strip(), None)
)

# Keep only rows with embeddings
df = df_excel[df_excel['embedding'].notna()].copy().reset_index(drop=True)
print(f"Loaded {len(df)} records with embeddings")

# ============================================================================
# STEP 2: Clean Text
# ============================================================================

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

# ============================================================================
# STEP 3: Remove Duplicates
# ============================================================================

print("Removing duplicates...")
df = df.drop_duplicates(subset=['bp_description_cleaned'], keep='first').reset_index(drop=True)
print(f"Unique incidents: {len(df)}")

# ============================================================================
# STEP 4: Prepare Embeddings
# ============================================================================

print("Preparing embeddings...")
descriptions = df['bp_description_cleaned'].tolist()

# Parse embeddings
if isinstance(df['embedding'].iloc[0], str):
    embeddings = np.array([eval(x) for x in df['embedding']])
elif isinstance(df['embedding'].iloc[0], list):
    embeddings = np.array(df['embedding'].tolist())
else:
    embeddings = np.vstack(df['embedding'].values)

embeddings_normalized = normalize(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# ============================================================================
# STEP 5: Configure UMAP for Dimensionality Reduction
# ============================================================================

print("Configuring UMAP...")

umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    min_dist=0.0,
    metric='cosine',
    low_memory=True,
    random_state=42
)

# ============================================================================
# STEP 6: Configure HDBSCAN for Clustering
# ============================================================================

print("Configuring HDBSCAN...")

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1
)

# ============================================================================
# STEP 7: Initialize and Fit BERTopic
# ============================================================================

print("Initializing BERTopic...")

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
    nr_topics='auto'
)

print("Fitting BERTopic model...")
topics, probabilities = topic_model.fit_transform(descriptions, embeddings_normalized)

# ============================================================================
# STEP 8: Get Topic Information
# ============================================================================

print("Extracting topic information...")

topic_info = topic_model.get_topic_info()
n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)

print(f"\nDiscovered {n_topics} topics")
print("\nTop 10 Topics:")
print(topic_info.head(11)[['Topic', 'Count', 'Name']])

# ============================================================================
# STEP 9: Add Results to DataFrame
# ============================================================================

print("\nAdding cluster assignments to data...")

df['cluster_id'] = topics
df['cluster_probability'] = probabilities

# Create topic names dictionary
topic_names = {}
for _, row in topic_info.iterrows():
    topic_id = row['Topic']
    topic_name = row['Name']
    # Extract keywords from name (format: "-1_keyword1_keyword2_keyword3")
    keywords = '_'.join(topic_name.split('_')[1:4]) if '_' in topic_name else topic_name
    topic_names[topic_id] = keywords

df['cluster_name'] = df['cluster_id'].map(topic_names)

# ============================================================================
# STEP 10: Create Cluster Statistics
# ============================================================================

print("Generating cluster statistics...")

cluster_stats = []
unique_topics = sorted([t for t in df['cluster_id'].unique() if t != -1])

for cluster_id in unique_topics:
    cluster_docs = df[df['cluster_id'] == cluster_id]
    
    # Get topic words
    topic_words = topic_model.get_topic(cluster_id)
    if topic_words:
        topic_keywords = ' | '.join([word for word, _ in topic_words[:10]])
    else:
        topic_keywords = 'N/A'
    
    stats = {
        'cluster_id': cluster_id,
        'cluster_name': topic_names.get(cluster_id, 'unknown'),
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

# Save topic info from BERTopic
topic_info.to_csv(output_dir / 'bertopic_topics.csv', index=False)

# Save all results
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
        
        print(f"Cluster {cluster_id} ({topic_name}): {len(cluster_df)} incidents")

# Save outliers (cluster_id = -1)
if (df['cluster_id'] == -1).any():
    outliers_df = df[df['cluster_id'] == -1]
    outliers_df.to_csv(output_dir / 'cluster_outliers.csv', index=False)
    print(f"\nOutliers: {len(outliers_df)} incidents")

# ============================================================================
# STEP 12: Display Summary
# ============================================================================

print("\n" + "="*80)
print("CLUSTERING COMPLETE")
print("="*80)
print(f"Total unique incidents: {len(df)}")
print(f"Number of clusters: {n_topics}")
print(f"Outliers: {(df['cluster_id'] == -1).sum()}")

print(f"\nTop 10 Largest Clusters:")
print(cluster_stats_df.nlargest(10, 'count')[
    ['cluster_id', 'cluster_name', 'count', 'avg_probability']
].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)

# ============================================================================
# STEP 13: Additional Analysis by Business Stream
# ============================================================================

print("\n" + "="*80)
print("CLUSTER DISTRIBUTION BY BUSINESS STREAM")
print("="*80)

for cluster_id in unique_topics[:5]:  # Show top 5 clusters
    cluster_data = df[df['cluster_id'] == cluster_id]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} ({topic_names.get(cluster_id, 'unknown')}):")
        business_stream_dist = cluster_data['business_stream_name'].value_counts().head(5)
        for stream, count in business_stream_dist.items():
            print(f"  - {stream}: {count} incidents ({count/len(cluster_data)*100:.1f}%)")

# ============================================================================
# STEP 14: Optional - Save Visualizations (if needed)
# ============================================================================

# Uncomment these lines if you want to generate visualizations
# print("\nGenerating visualizations...")
# topic_model.visualize_topics().write_html(output_dir / 'topic_visualization.html')
# topic_model.visualize_barchart(top_n_topics=20).write_html(output_dir / 'topic_barchart.html')
# topic_model.visualize_hierarchy().write_html(output_dir / 'topic_hierarchy.html')

print("\nAll done!")
