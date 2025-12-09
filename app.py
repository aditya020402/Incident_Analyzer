import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
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

# Join Excel and Parquet on incident_id
df = pd.merge(df_excel, df_parquet, on='incident_id', how='inner')
print(f"After joining: {len(df)} records with embeddings")

# ============================================================================
# STEP 2: Clean bp_description Text
# ============================================================================

print("\nCleaning bp_description field...")

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Apply cleaning
df['bp_description_cleaned'] = df['bp_description'].apply(clean_text)

# Remove empty descriptions after cleaning
df_before_empty = len(df)
df = df[df['bp_description_cleaned'].str.len() > 0]
print(f"Removed {df_before_empty - len(df)} records with empty descriptions")

# ============================================================================
# STEP 3: Remove Duplicates Based on Description
# ============================================================================

print("\nRemoving duplicate descriptions...")

# Count duplicates before removal
duplicate_count = df.duplicated(subset=['bp_description_cleaned'], keep='first').sum()
print(f"Found {duplicate_count} duplicate descriptions")

# Keep first occurrence of each unique description
df_before_dedup = len(df)
df = df.drop_duplicates(subset=['bp_description_cleaned'], keep='first')
print(f"Removed {df_before_dedup - len(df)} duplicate incidents")
print(f"Remaining records: {len(df)}")

# Reset index
df = df.reset_index(drop=True)

# ============================================================================
# STEP 4: Prepare Embeddings
# ============================================================================

print("\nPreparing embeddings...")

# Extract descriptions (use cleaned version)
descriptions = df['bp_description_cleaned'].tolist()

# Parse embeddings
if isinstance(df['embedding'].iloc[0], str):
    # If stored as string like "[0.1, 0.2, ...]"
    embeddings = np.array([eval(x) for x in df['embedding']])
else:
    # If already in array format
    embeddings = np.array(df['embedding'].tolist())

# Normalize embeddings
embeddings_normalized = normalize(embeddings)

print(f"Processing {len(descriptions)} unique incidents")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# ============================================================================
# STEP 5: Preprocess Text for CTM
# ============================================================================

print("\nPreprocessing text for topic modeling...")

sp = WhiteSpacePreprocessing(descriptions, stopwords_language='english')
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

print(f"Vocabulary size: {len(vocab)}")
print(f"Documents after preprocessing: {len(preprocessed_documents)}")

# Align embeddings with retained documents
embeddings_aligned = embeddings_normalized[retained_indices]

# ============================================================================
# STEP 6: Find Optimal Number of Topics
# ============================================================================

print("\nFinding optimal number of topics...")

tp = TopicModelDataPreparation("english")
training_dataset = tp.fit(
    text_for_contextual=unpreprocessed_corpus,
    text_for_bow=preprocessed_documents,
    custom_embeddings=embeddings_aligned
)

best_score = -1
best_n_topics = 20
topic_range = range(20, 101, 20)  # Test 20, 40, 60, 80, 100

for n_topics in topic_range:
    print(f"\nTesting {n_topics} topics...")
    
    ctm = ZeroShotTM(
        bow_size=len(tp.vocab),
        contextual_size=embeddings_aligned.shape[1],
        n_components=n_topics,
        num_epochs=20,
        hidden_sizes=(100,),
        lr=2e-3
    )
    
    ctm.fit(training_dataset)
    
    topic_document_matrix = ctm.get_doc_topic_distribution(training_dataset)
    topic_labels = np.argmax(topic_document_matrix, axis=1)
    
    sil_score = silhouette_score(
        embeddings_aligned, 
        topic_labels, 
        sample_size=min(10000, len(topic_labels))
    )
    
    print(f"Silhouette Score: {sil_score:.4f}")
    
    if sil_score > best_score:
        best_score = sil_score
        best_n_topics = n_topics

print(f"\nOptimal number of topics: {best_n_topics} (Silhouette: {best_score:.4f})")

# ============================================================================
# STEP 7: Train Final Model
# ============================================================================

print(f"\nTraining final model with {best_n_topics} topics...")

ctm_final = ZeroShotTM(
    bow_size=len(tp.vocab),
    contextual_size=embeddings_aligned.shape[1],
    n_components=best_n_topics,
    num_epochs=50,
    hidden_sizes=(100,),
    lr=2e-3,
    batch_size=256,
    dropout=0.1
)

ctm_final.fit(training_dataset)

# ============================================================================
# STEP 8: Extract Topics and Assign Clusters
# ============================================================================

print("\nExtracting topics and assigning clusters...")

# Get topic words
topics = ctm_final.get_topics(10)

# Display topics
print("\n" + "="*80)
print("DISCOVERED TOPICS:")
print("="*80)
for topic_id, words in enumerate(topics):
    print(f"Topic {topic_id}: {' | '.join(words)}")

# Get document-topic distribution and assign clusters
topic_document_matrix = ctm_final.get_doc_topic_distribution(training_dataset)
cluster_labels = np.argmax(topic_document_matrix, axis=1)
cluster_probabilities = np.max(topic_document_matrix, axis=1)

# ============================================================================
# STEP 9: Add Results to DataFrame
# ============================================================================

# Create results for retained documents
df_retained = df.iloc[retained_indices].copy()
df_retained['cluster_id'] = cluster_labels
df_retained['cluster_probability'] = cluster_probabilities

# Create topic labels
topic_names = {}
for topic_id, words in enumerate(topics):
    topic_names[topic_id] = '_'.join(words[:3])

df_retained['cluster_name'] = df_retained['cluster_id'].map(topic_names)

# Handle filtered documents
filtered_indices_list = list(set(range(len(df))) - set(retained_indices))
if filtered_indices_list:
    df_filtered = df.iloc[filtered_indices_list].copy()
    df_filtered['cluster_id'] = -1
    df_filtered['cluster_probability'] = 0.0
    df_filtered['cluster_name'] = 'filtered'
    df_final = pd.concat([df_retained, df_filtered]).sort_index()
else:
    df_final = df_retained

# Ensure column order: original fields + cleaned description + cluster fields
column_order = [
    'incident_id',
    'alert_count',
    'bp_description',
    'bp_description_cleaned',
    'bp_impacted_service',
    'bp_isacswci',
    'business_stream_name',
    'embedding',
    'cluster_id',
    'cluster_probability',
    'cluster_name'
]
df_final = df_final[column_order]

# ============================================================================
# STEP 10: Create Cluster Statistics
# ============================================================================

print("\nGenerating cluster statistics...")

cluster_stats = []
for cluster_id in range(best_n_topics):
    cluster_docs = df_final[df_final['cluster_id'] == cluster_id]
    
    if len(cluster_docs) > 0:
        stats = {
            'cluster_id': cluster_id,
            'cluster_name': topic_names[cluster_id],
            'topic_words': ' | '.join(topics[cluster_id]),
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

# Save cleaning and deduplication report
cleaning_report = {
    'original_excel_records': len(df_excel),
    'records_with_embeddings': len(df_excel) + len(df_parquet) - len(df),
    'records_after_join': len(df_excel),
    'records_after_cleaning': df_before_empty,
    'records_after_dedup': len(df),
    'duplicates_removed': df_before_dedup - len(df),
    'empty_descriptions_removed': df_before_empty - df_before_dedup,
    'final_clustered_records': len(df_retained),
    'filtered_records': len(filtered_indices_list),
    'total_clusters': best_n_topics
}

pd.DataFrame([cleaning_report]).to_csv(output_dir / 'cleaning_report.csv', index=False)

# Save cluster summary
cluster_stats_df.to_csv(output_dir / 'cluster_summary.csv', index=False)

# Save complete results with all original fields
df_final.to_csv(output_dir / 'all_incidents_clustered.csv', index=False)

# Save individual cluster files with all original fields
print("\nCreating individual cluster files...")
for cluster_id in range(best_n_topics):
    cluster_df = df_final[df_final['cluster_id'] == cluster_id]
    
    if len(cluster_df) > 0:
        topic_name = topic_names[cluster_id]
        # Clean topic name for filename
        clean_topic_name = topic_name.replace(' ', '_').replace('/', '_')[:50]
        filename = f'cluster_{cluster_id}_{clean_topic_name}.csv'
        cluster_df.to_csv(output_dir / filename, index=False)
        
        print(f"Cluster {cluster_id} ({topic_name}): {len(cluster_df)} incidents")
        print(f"  - Avg Alert Count: {cluster_df['alert_count'].mean():.2f}")
        print(f"  - Unique Services: {cluster_df['bp_impacted_service'].nunique()}")
        print(f"  - Unique Business Streams: {cluster_df['business_stream_name'].nunique()}")

# Save filtered documents if any
if (df_final['cluster_id'] == -1).any():
    filtered_df = df_final[df_final['cluster_id'] == -1]
    filtered_df.to_csv(output_dir / 'cluster_filtered.csv', index=False)
    print(f"\nFiltered: {len(filtered_df)} incidents")

# ============================================================================
# STEP 12: Display Summary
# ============================================================================

print("\n" + "="*80)
print("CLUSTERING COMPLETE")
print("="*80)
print(f"Original Excel records: {len(df_excel)}")
print(f"Records after join with embeddings: {len(df_excel)}")
print(f"Duplicates removed: {duplicate_count}")
print(f"Empty descriptions removed: {df_before_empty - len(df)}")
print(f"Final unique incidents: {len(df)}")
print(f"Successfully clustered: {len(df_retained)}")
print(f"Filtered during preprocessing: {len(filtered_indices_list)}")
print(f"Number of clusters: {best_n_topics}")

print(f"\nTop 10 Largest Clusters:")
print(cluster_stats_df.nlargest(10, 'count')[
    ['cluster_id', 'cluster_name', 'count', 'avg_probability', 'unique_services']
].to_string(index=False))

print(f"\nResults saved to: {output_dir}/")
print("="*80)

# ============================================================================
# STEP 13: Additional Analysis
# ============================================================================

print("\n" + "="*80)
print("CLUSTER DISTRIBUTION BY BUSINESS STREAM")
print("="*80)

for cluster_id in range(min(5, best_n_topics)):  # Show top 5 clusters
    cluster_data = df_final[df_final['cluster_id'] == cluster_id]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} ({topic_names[cluster_id]}):")
        business_stream_dist = cluster_data['business_stream_name'].value_counts().head(5)
        for stream, count in business_stream_dist.items():
            print(f"  - {stream}: {count} incidents ({count/len(cluster_data)*100:.1f}%)")
