import json
nb = json.load(open('/root/Machine Learning/homework1-kmeans.ipynb', encoding='utf-8'))
print('total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    t = c['cell_type']
    src = c['source']
    blob = ''.join(src) if isinstance(src, list) else src
    first = blob.strip().split('\n')[0][:70] if blob.strip() else '(empty)'
    print(f'{i:>2} [{t}] {first}')
