from sys import argv

print("digraph G {")
# print("graph [ overlap=false ]")
with open(argv[1]) as f:
    next(f)
    for line in f:
        if line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        progenitor = int(cols[1])
        descendant = int(cols[3])
        if descendant == -1:
            continue
        print(f"  {progenitor} -> {descendant};")
print("}")
