import re

# Read the file
with open('Dataset/twitter/gnn_pipeline.py', 'r') as f:
    content = f.read()

# Find and replace the broken hashtag handling code
old_code = '''        # Edge 3: post -> hashtag (uses_hashtag)
        hashtag_edges = []
        if 'hashtags' in df.columns:
            for i, hlist in enumerate(df['hashtags']):
                if pd.notna(hlist):
                    if isinstance(hlist, str):
                        hlist = hlist.split()
                    for h in hlist:
                        h_idx = self._get_hashtag_idx(h)
                        if h_idx is not None:
                            hashtag_edges.append((i, h_idx))
        edges[('post','uses_hashtag','hashtag')] = hashtag_edges'''

new_code = '''        # Edge 3: post -> hashtag (uses_hashtag) - FIXED PROPERLY
        hashtag_edges = []
        if 'hashtags' in df.columns:
            for i, hlist in enumerate(df['hashtags']):
                # ✅ Use the _parse_hashtags() method (NOT pd.notna)
                hashtags = self._parse_hashtags(hlist)
                
                for h in hashtags:
                    # Skip empty strings
                    if isinstance(h, str) and len(h.strip()) == 0:
                        continue
                    
                    h_idx = self._get_hashtag_idx(str(h))
                    if h_idx is not None:
                        hashtag_edges.append((i, h_idx))

        edges[('post','uses_hashtag','hashtag')] = hashtag_edges'''

# Replace
if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Successfully replaced the broken code!")
else:
    print("❌ Could not find the exact old code")
    exit(1)

# Write back
with open('Dataset/twitter/gnn_pipeline.py', 'w') as f:
    f.write(content)

print("✅ File updated successfully!")
