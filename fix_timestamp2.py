# Read the file
with open('Dataset/twitter/gnn_pipeline.py', 'r') as f:
    content = f.read()

# Find and replace with proper syntax
old_code = """        collab_edges = []
        if 'timestamp' in df.columns:
            # Safe timestamp parsing with error handling
            try:
                dfc['hour'] = pd.to_datetime(dfc['timestamp'], format='%a %b %d %H:%M:%S %z %Y').dt.floor('H')
            except:
                try:
                    dfc['hour'] = pd.to_datetime(dfc['timestamp'], infer_datetime_format=True).dt.floor('H')
                except:
                    # If parsing fails, just skip user_collab edges
                    continue
            for _, g in dfc.groupby('hour'):"""

new_code = """        collab_edges = []
        if 'timestamp' in df.columns:
            dfc = df.copy()
            # Safe timestamp parsing with error handling
            try:
                dfc['hour'] = pd.to_datetime(dfc['timestamp'], format='%a %b %d %H:%M:%S %z %Y').dt.floor('H')
            except:
                try:
                    dfc['hour'] = pd.to_datetime(dfc['timestamp'], infer_datetime_format=True).dt.floor('H')
                except:
                    # If parsing fails, just skip user_collab edges
                    dfc = None
            
            if dfc is not None:
                for _, g in dfc.groupby('hour'):"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Fixed timestamp parsing!")
    
    # Also need to indent the rest of the loop
    content = content.replace(
        "                users = g['user_id'].unique()",
        "                    users = g['user_id'].unique()"
    )
    content = content.replace(
        "                for i, u1 in enumerate(users):",
        "                    for i, u1 in enumerate(users):"
    )
    content = content.replace(
        "                    for u2 in users[i+1:]:",
        "                        for u2 in users[i+1:]:"
    )
    content = content.replace(
        "                        collab_edges.append((self._get_user_idx(u1), self._get_user_idx(u2)))",
        "                            collab_edges.append((self._get_user_idx(u1), self._get_user_idx(u2)))"
    )
else:
    print("❌ Could not find code section")
    exit(1)

with open('Dataset/twitter/gnn_pipeline.py', 'w') as f:
    f.write(content)

print("✅ File updated!")
