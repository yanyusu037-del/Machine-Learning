import json
nb = json.load(open('/root/Machine Learning/homework1-kmeans.ipynb', encoding='utf-8'))

def src_of(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

for i, c in enumerate(nb['cells']):
    print(f"\n{'='*70}\n### CELL {i} [{c['cell_type']}]\n{'='*70}")
    print(src_of(c)[:2500])
