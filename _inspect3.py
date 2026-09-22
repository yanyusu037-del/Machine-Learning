import inspect
from sklearn.cluster._kmeans import KMeans, k_means, _labels_inertia

src_fit = inspect.getsource(KMeans.fit)
print("=== fit source length ===")
print("chars:", len(src_fit), "| lines:", src_fit.count(chr(10))+1)

# _labels_inertia (shown in cell 18)
src_li = inspect.getsource(_labels_inertia)
print("\n=== _labels_inertia length ===")
print("chars:", len(src_li), "| lines:", src_li.count(chr(10))+1)

# predict/transform/score source lengths
for name in ["predict", "transform", "score"]:
    s = inspect.getsource(getattr(KMeans, name))
    print(f"=== {name} === chars: {len(s)}, lines: {s.count(chr(10))+1}")

# k_means source
s = inspect.getsource(k_means)
print(f"=== k_means === chars: {len(s)}, lines: {s.count(chr(10))+1}")
