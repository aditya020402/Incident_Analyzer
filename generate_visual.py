import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: Load Clustered Data
# ============================================================================

print("Loading clustered data...")
df = pd.read_csv('clusters/all_incidents_clustered.csv')

# Get unique clusters (excluding outliers -1)
unique_clusters = sorted([c for c in df['cluster_id'].unique() if c != -1])
n_clusters = len(unique_clusters)

print(f"Total incidents: {len(df)}")
print(f"Number of clusters: {n_clusters}")
print(f"Outliers: {(df['cluster_id'] == -1).sum()}")

# ============================================================================
# STEP 2: Overall Statistics
# ============================================================================

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

overall_stats = {
    'total_incidents': len(df),
    'total_clusters': n_clusters,
    'outliers': (df['cluster_id'] == -1).sum(),
    'unique_services': df['bp_impacted_service'].nunique(),
    'unique_business_streams': df['business_stream_name'].nunique(),
    'total_alerts': df['alert_count'].sum(),
    'avg_alerts_per_incident': df['alert_count'].mean()
}

print(f"Total Incidents: {overall_stats['total_incidents']}")
print(f"Total Clusters: {overall_stats['total_clusters']}")
print(f"Outliers: {overall_stats['outliers']}")
print(f"Unique Services: {overall_stats['unique_services']}")
print(f"Unique Business Streams: {overall_stats['unique_business_streams']}")
print(f"Average Alerts per Incident: {overall_stats['avg_alerts_per_incident']:.2f}")

# ============================================================================
# STEP 3: Detailed Cluster Statistics
# ============================================================================

print("\n" + "="*80)
print("DETAILED CLUSTER STATISTICS")
print("="*80)

output_dir = Path('cluster_analysis')
output_dir.mkdir(exist_ok=True)

all_cluster_stats = []

for cluster_id in unique_clusters:
    cluster_df = df[df['cluster_id'] == cluster_id]
    cluster_name = cluster_df['cluster_name'].iloc[0] if len(cluster_df) > 0 else f'cluster_{cluster_id}'
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'='*80}")
    print(f"Total Incidents: {len(cluster_df)}")
    print(f"Percentage of Total: {len(cluster_df)/len(df)*100:.2f}%")
    print(f"Average Probability: {cluster_df['cluster_probability'].mean():.4f}")
    
    # Business Stream Statistics
    print(f"\n--- Business Stream Distribution ---")
    business_stream_dist = cluster_df['business_stream_name'].value_counts()
    for stream, count in business_stream_dist.head(10).items():
        print(f"  {stream}: {count} ({count/len(cluster_df)*100:.1f}%)")
    
    # Service Statistics
    print(f"\n--- Service Distribution ---")
    service_dist = cluster_df['bp_impacted_service'].value_counts()
    for service, count in service_dist.head(10).items():
        print(f"  {service}: {count} ({count/len(cluster_df)*100:.1f}%)")
    
    # Alert Statistics
    print(f"\n--- Alert Statistics ---")
    print(f"  Total Alerts: {cluster_df['alert_count'].sum()}")
    print(f"  Average Alerts: {cluster_df['alert_count'].mean():.2f}")
    print(f"  Min Alerts: {cluster_df['alert_count'].min()}")
    print(f"  Max Alerts: {cluster_df['alert_count'].max()}")
    
    # ISACSWCI Statistics (if it indicates automation)
    print(f"\n--- ISACSWCI Distribution ---")
    isacswci_dist = cluster_df['bp_isacswci'].value_counts()
    for value, count in isacswci_dist.items():
        print(f"  {value}: {count} ({count/len(cluster_df)*100:.1f}%)")
    
    # Store detailed stats for this cluster
    cluster_stat = {
        'cluster_id': cluster_id,
        'cluster_name': cluster_name,
        'total_incidents': len(cluster_df),
        'percentage_of_total': len(cluster_df)/len(df)*100,
        'avg_probability': cluster_df['cluster_probability'].mean(),
        'total_alerts': cluster_df['alert_count'].sum(),
        'avg_alerts': cluster_df['alert_count'].mean(),
        'min_alerts': cluster_df['alert_count'].min(),
        'max_alerts': cluster_df['alert_count'].max(),
        'unique_business_streams': cluster_df['business_stream_name'].nunique(),
        'unique_services': cluster_df['bp_impacted_service'].nunique(),
        'top_business_stream': business_stream_dist.index[0] if len(business_stream_dist) > 0 else 'N/A',
        'top_business_stream_count': business_stream_dist.iloc[0] if len(business_stream_dist) > 0 else 0,
        'top_service': service_dist.index[0] if len(service_dist) > 0 else 'N/A',
        'top_service_count': service_dist.iloc[0] if len(service_dist) > 0 else 0
    }
    all_cluster_stats.append(cluster_stat)
    
    # Save detailed breakdown for this cluster
    cluster_detail = pd.DataFrame({
        'metric': ['Total Incidents', 'Avg Probability', 'Total Alerts', 'Avg Alerts',
                   'Unique Business Streams', 'Unique Services'],
        'value': [len(cluster_df), cluster_df['cluster_probability'].mean(),
                 cluster_df['alert_count'].sum(), cluster_df['alert_count'].mean(),
                 cluster_df['business_stream_name'].nunique(), cluster_df['bp_impacted_service'].nunique()]
    })
    
    # Business stream breakdown
    bs_breakdown = business_stream_dist.reset_index()
    bs_breakdown.columns = ['business_stream', 'count']
    bs_breakdown['percentage'] = (bs_breakdown['count'] / len(cluster_df) * 100).round(2)
    
    # Service breakdown
    service_breakdown = service_dist.reset_index()
    service_breakdown.columns = ['service', 'count']
    service_breakdown['percentage'] = (service_breakdown['count'] / len(cluster_df) * 100).round(2)
    
    # Save to Excel with multiple sheets
    excel_path = output_dir / f'cluster_{cluster_id}_detailed_stats.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        cluster_detail.to_excel(writer, sheet_name='Overview', index=False)
        bs_breakdown.to_excel(writer, sheet_name='Business_Streams', index=False)
        service_breakdown.to_excel(writer, sheet_name='Services', index=False)
        cluster_df.to_excel(writer, sheet_name='All_Incidents', index=False)

# Save summary of all clusters
all_cluster_stats_df = pd.DataFrame(all_cluster_stats)
all_cluster_stats_df.to_csv(output_dir / 'all_clusters_summary.csv', index=False)

# ============================================================================
# STEP 4: Create Visualizations
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

# 1. Cluster Size Distribution
plt.figure(figsize=(14, 6))
cluster_sizes = all_cluster_stats_df.sort_values('total_incidents', ascending=False)
plt.barh(range(len(cluster_sizes)), cluster_sizes['total_incidents'])
plt.yticks(range(len(cluster_sizes)), 
           [f"C{row['cluster_id']}: {row['cluster_name'][:30]}" for _, row in cluster_sizes.iterrows()],
           fontsize=8)
plt.xlabel('Number of Incidents')
plt.title('Cluster Size Distribution')
plt.tight_layout()
plt.savefig(viz_dir / 'cluster_sizes.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Alert Distribution by Cluster
plt.figure(figsize=(14, 6))
cluster_alerts = all_cluster_stats_df.sort_values('avg_alerts', ascending=False)
plt.barh(range(len(cluster_alerts)), cluster_alerts['avg_alerts'])
plt.yticks(range(len(cluster_alerts)), 
           [f"C{row['cluster_id']}: {row['cluster_name'][:30]}" for _, row in cluster_alerts.iterrows()],
           fontsize=8)
plt.xlabel('Average Alert Count')
plt.title('Average Alerts per Cluster')
plt.tight_layout()
plt.savefig(viz_dir / 'cluster_avg_alerts.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Top 10 Clusters by Size (Pie Chart)
plt.figure(figsize=(10, 10))
top_10_clusters = all_cluster_stats_df.nlargest(10, 'total_incidents')
others_count = df[~df['cluster_id'].isin(top_10_clusters['cluster_id'])]['cluster_id'].count()
labels = [f"C{row['cluster_id']}: {row['cluster_name'][:20]}" for _, row in top_10_clusters.iterrows()]
sizes = top_10_clusters['total_incidents'].tolist()
if others_count > 0:
    labels.append('Others')
    sizes.append(others_count)
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Clusters by Incident Count')
plt.tight_layout()
plt.savefig(viz_dir / 'top_10_clusters_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Business Stream Distribution Across All Clusters
plt.figure(figsize=(14, 8))
top_streams = df['business_stream_name'].value_counts().head(15)
plt.bar(range(len(top_streams)), top_streams.values)
plt.xticks(range(len(top_streams)), top_streams.index, rotation=45, ha='right')
plt.ylabel('Number of Incidents')
plt.title('Top 15 Business Streams (All Clusters)')
plt.tight_layout()
plt.savefig(viz_dir / 'top_business_streams.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Service Distribution Across All Clusters
plt.figure(figsize=(14, 8))
top_services = df['bp_impacted_service'].value_counts().head(15)
plt.bar(range(len(top_services)), top_services.values)
plt.xticks(range(len(top_services)), top_services.index, rotation=45, ha='right')
plt.ylabel('Number of Incidents')
plt.title('Top 15 Impacted Services (All Clusters)')
plt.tight_layout()
plt.savefig(viz_dir / 'top_services.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Heatmap: Top Clusters vs Top Business Streams
top_10_clusters_ids = all_cluster_stats_df.nlargest(10, 'total_incidents')['cluster_id'].tolist()
top_10_streams = df['business_stream_name'].value_counts().head(10).index.tolist()

heatmap_data = []
for cluster_id in top_10_clusters_ids:
    cluster_df = df[df['cluster_id'] == cluster_id]
    row = []
    for stream in top_10_streams:
        count = len(cluster_df[cluster_df['business_stream_name'] == stream])
        row.append(count)
    heatmap_data.append(row)

plt.figure(figsize=(12, 8))
cluster_labels = [f"C{cid}: {all_cluster_stats_df[all_cluster_stats_df['cluster_id']==cid]['cluster_name'].iloc[0][:15]}" 
                  for cid in top_10_clusters_ids]
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[s[:20] for s in top_10_streams],
            yticklabels=cluster_labels)
plt.title('Cluster vs Business Stream Heatmap (Top 10 Each)')
plt.xlabel('Business Stream')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig(viz_dir / 'cluster_stream_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. ISACSWCI Distribution by Cluster (if applicable)
if 'bp_isacswci' in df.columns:
    plt.figure(figsize=(14, 8))
    isacswci_by_cluster = df.groupby(['cluster_id', 'bp_isacswci']).size().unstack(fill_value=0)
    isacswci_by_cluster = isacswci_by_cluster.loc[unique_clusters[:10]]  # Top 10 clusters
    isacswci_by_cluster.plot(kind='bar', stacked=True)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Incidents')
    plt.title('ISACSWCI Distribution by Cluster (Top 10)')
    plt.legend(title='ISACSWCI', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(viz_dir / 'isacswci_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Alert Count Distribution (Histogram)
plt.figure(figsize=(12, 6))
plt.hist(df['alert_count'], bins=50, edgecolor='black')
plt.xlabel('Alert Count')
plt.ylabel('Number of Incidents')
plt.title('Alert Count Distribution Across All Incidents')
plt.tight_layout()
plt.savefig(viz_dir / 'alert_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Cluster Probability Distribution
plt.figure(figsize=(12, 6))
plt.hist(df[df['cluster_id'] != -1]['cluster_probability'], bins=50, edgecolor='black')
plt.xlabel('Cluster Probability')
plt.ylabel('Number of Incidents')
plt.title('Cluster Assignment Probability Distribution')
plt.tight_layout()
plt.savefig(viz_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 5: Create Department-wise Analysis for Each Cluster
# ============================================================================

print("\n" + "="*80)
print("CREATING DEPARTMENT-WISE ANALYSIS")
print("="*80)

dept_dir = output_dir / 'department_analysis'
dept_dir.mkdir(exist_ok=True)

for cluster_id in unique_clusters:
    cluster_df = df[df['cluster_id'] == cluster_id]
    cluster_name = cluster_df['cluster_name'].iloc[0] if len(cluster_df) > 0 else f'cluster_{cluster_id}'
    
    # Department (Business Stream) breakdown
    dept_stats = []
    for dept in cluster_df['business_stream_name'].unique():
        dept_df = cluster_df[cluster_df['business_stream_name'] == dept]
        
        dept_stat = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'business_stream': dept,
            'incident_count': len(dept_df),
            'percentage_in_cluster': len(dept_df)/len(cluster_df)*100,
            'total_alerts': dept_df['alert_count'].sum(),
            'avg_alerts': dept_df['alert_count'].mean(),
            'unique_services': dept_df['bp_impacted_service'].nunique(),
            'avg_probability': dept_df['cluster_probability'].mean()
        }
        dept_stats.append(dept_stat)
    
    dept_stats_df = pd.DataFrame(dept_stats).sort_values('incident_count', ascending=False)
    dept_stats_df.to_csv(dept_dir / f'cluster_{cluster_id}_department_breakdown.csv', index=False)
    
    # Create visualization for this cluster's departments
    if len(dept_stats_df) > 1:
        plt.figure(figsize=(12, 6))
        top_depts = dept_stats_df.head(10)
        plt.bar(range(len(top_depts)), top_depts['incident_count'])
        plt.xticks(range(len(top_depts)), top_depts['business_stream'], rotation=45, ha='right')
        plt.ylabel('Number of Incidents')
        plt.title(f'Cluster {cluster_id}: {cluster_name[:40]} - Department Distribution')
        plt.tight_layout()
        plt.savefig(viz_dir / f'cluster_{cluster_id}_departments.png', dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nAnalysis complete! Results saved to: {output_dir}/")
print(f"Visualizations saved to: {viz_dir}/")
print(f"Department analysis saved to: {dept_dir}/")

# ============================================================================
# STEP 6: Generate Summary Report
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY REPORT")
print("="*80)

summary_report = []
summary_report.append("="*80)
summary_report.append("CLUSTER ANALYSIS SUMMARY REPORT")
summary_report.append("="*80)
summary_report.append(f"\nTotal Incidents Analyzed: {len(df)}")
summary_report.append(f"Total Clusters Found: {n_clusters}")
summary_report.append(f"Outliers: {(df['cluster_id'] == -1).sum()}")
summary_report.append(f"\n{'='*80}")
summary_report.append("TOP 10 CLUSTERS BY SIZE")
summary_report.append("="*80)

for idx, (_, row) in enumerate(all_cluster_stats_df.nlargest(10, 'total_incidents').iterrows(), 1):
    summary_report.append(f"\n{idx}. Cluster {row['cluster_id']}: {row['cluster_name']}")
    summary_report.append(f"   Incidents: {row['total_incidents']} ({row['percentage_of_total']:.2f}%)")
    summary_report.append(f"   Avg Alerts: {row['avg_alerts']:.2f}")
    summary_report.append(f"   Top Business Stream: {row['top_business_stream']} ({row['top_business_stream_count']} incidents)")
    summary_report.append(f"   Top Service: {row['top_service']} ({row['top_service_count']} incidents)")

summary_report.append(f"\n{'='*80}")
summary_report.append("FILES GENERATED")
summary_report.append("="*80)
summary_report.append(f"- All clusters summary: {output_dir}/all_clusters_summary.csv")
summary_report.append(f"- Individual cluster details: {output_dir}/cluster_*_detailed_stats.xlsx")
summary_report.append(f"- Visualizations: {viz_dir}/*.png")
summary_report.append(f"- Department analysis: {dept_dir}/cluster_*_department_breakdown.csv")

# Save summary report
with open(output_dir / 'summary_report.txt', 'w') as f:
    f.write('\n'.join(summary_report))

# Print summary
print('\n'.join(summary_report))
