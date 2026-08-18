import os
base = r"algorithms"

# Fix kmeans.py - replace the broken float("inf") line
with open(os.path.join(base, "kmeans.py"), "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('float(chr(34)+chr(105)+chr(110)+chr(102)+chr(34))', 'float("inf")')
with open(os.path.join(base, "kmeans.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed kmeans.py")
