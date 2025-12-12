# Sort by incident_id to ensure consistent ordering
df = df.sort_values('incident_id').reset_index(drop=True)

# Group by cleaned description
duplicate_groups = df.groupby('bp_description_cleaned').agg({
    'incident_id': lambda x: list(x),
    'alert_count': 'first',
    'bp_description': 'first',
    'bp_impacted_service': 'first',
    'bp_isacswci': 'first',
    'business_stream_name': 'first',
    'embedding': 'first'
}).reset_index()

# Add duplicate count and incident_id array
duplicate_groups['duplicate_count'] = duplicate_groups['incident_id'].apply(len)
duplicate_groups['duplicate_incident_ids'] = duplicate_groups['incident_id'].apply(lambda x: x)
duplicate_groups['incident_id'] = duplicate_groups['incident_id'].apply(lambda x: x[0])  # Keep first as primary
