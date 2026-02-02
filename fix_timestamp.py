# Read the file
with open('Dataset/twitter/gnn_pipeline.py', 'r') as f:
    content = f.read()

# Find the problematic line
old_line = "            dfc['hour'] = pd.to_datetime(dfc['timestamp']).dt.floor('H')"

new_code = """            # Safe timestamp parsing with error handling
            try:
                dfc['hour'] = pd.to_datetime(dfc['timestamp'], format='%a %b %d %H:%M:%S %z %Y').dt.floor('H')
            except:
                try:
                    dfc['hour'] = pd.to_datetime(dfc['timestamp'], infer_datetime_format=True).dt.floor('H')
                except:
                    # If parsing fails, just skip user_collab edges
                    continue"""

if old_line in content:
    content = content.replace(old_line, new_code)
    print("✅ Successfully fixed timestamp parsing!")
else:
    print("❌ Could not find the timestamp line")
    exit(1)

# Write back
with open('Dataset/twitter/gnn_pipeline.py', 'w') as f:
    f.write(content)

print("✅ File updated successfully!")
