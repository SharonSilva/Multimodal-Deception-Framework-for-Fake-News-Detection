# Read the file
with open('Dataset/twitter/gnn_pipeline.py', 'r') as f:
    content = f.read()

# The method to insert (after _get_cluster_idx)
method_code = '''    
    def _parse_hashtags(self, hlist):
        """
        Safely parse hashtag list from various formats.
        Returns list of hashtags or empty list if parsing fails.
        
        Handles:
        - None values
        - NaN/float NaN
        - Strings (splits by whitespace)
        - Lists/tuples/sets
        - NumPy arrays
        """
        # Handle None
        if hlist is None:
            return []
        
        # Handle NaN/float NaN (scalar check)
        if isinstance(hlist, float):
            if pd.isna(hlist):
                return []
            return []
        
        # Handle string - split by whitespace
        if isinstance(hlist, str):
            if len(hlist.strip()) == 0:
                return []
            return hlist.split()
        
        # Handle list/tuple/set
        if isinstance(hlist, (list, tuple, set)):
            return list(hlist)
        
        # Handle numpy arrays
        if isinstance(hlist, np.ndarray):
            return hlist.tolist()
        
        # Fallback: return empty
        return []
'''

# Find where to insert it - after _get_cluster_idx method
insert_marker = "        return self.cluster_id_map[c]"

if insert_marker in content:
    # Find the position
    pos = content.find(insert_marker)
    if pos != -1:
        # Find the end of that line
        end_pos = content.find('\n', pos)
        # Insert the new method after this line
        new_content = content[:end_pos+1] + method_code + content[end_pos+1:]
        
        # Write back
        with open('Dataset/twitter/gnn_pipeline.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully added _parse_hashtags method!")
    else:
        print("❌ Could not find insertion point")
        exit(1)
else:
    print("❌ Could not find _get_cluster_idx method")
    exit(1)
