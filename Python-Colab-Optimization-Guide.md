# Scaling Python in Google Colab: Optimization Guide

A practical guide for handling 10,000+ data points efficiently in Google Colab.

---

## 1. Profile Before You Optimize

Find the actual bottleneck before changing anything.

```python
# Time an entire cell
%%time

# Benchmark a cell (runs multiple times)
%%timeit

# Line-by-line profiling
!pip install line_profiler
%load_ext line_profiler
%lprun -f your_function your_function(args)

# Memory profiling
!pip install memory_profiler
%load_ext memory_profiler
%memit your_function(args)
```

---

## 2. Eliminate Python Loops (Vectorization)

**This is the single biggest performance win.** Replace row-by-row `for` loops with vectorized operations.

### Bad (slow)
```python
results = []
for i, row in df.iterrows():
    results.append(row['price'] * row['quantity'])
df['total'] = results
```

### Good (fast)
```python
df['total'] = df['price'] * df['quantity']
```

### Key Rules
- Avoid `iterrows()` and `itertuples()` when possible
- Use built-in pandas methods: `.str`, `.dt`, `.apply()` (sparingly)
- Use NumPy array operations instead of element-wise loops

---

## 3. Optimize Data Types and Memory

### Downcast Numeric Columns
```python
# Reduce float64 to float32 (halves memory)
df['price'] = pd.to_numeric(df['price'], downcast='float')
df['quantity'] = pd.to_numeric(df['quantity'], downcast='integer')
```

### Use Category for Repeated Strings
```python
# If a column has limited unique values (e.g., product categories)
df['category'] = df['category'].astype('category')
# Can reduce memory by 90%+ for string columns
```

### Check Memory Usage
```python
df.info(memory_usage='deep')
```

---

## 4. Efficient File I/O

### Switch from CSV to Parquet
```python
# Save as Parquet (smaller, faster, preserves types)
df.to_parquet('data.parquet')

# Read Parquet (much faster than CSV)
df = pd.read_parquet('data.parquet')
```

### Only Load What You Need
```python
# Select specific columns
df = pd.read_csv('data.csv', usecols=['product', 'price', 'quantity'])

# Specify dtypes upfront (avoids inference overhead)
df = pd.read_csv('data.csv', dtype={'product': 'category', 'price': 'float32'})
```

### Chunked Processing for Large Files
```python
results = []
for chunk in pd.read_csv('big_file.csv', chunksize=10_000):
    processed = process(chunk)
    results.append(processed)
df = pd.concat(results)
```

---

## 5. Faster Library Alternatives

| Instead of | Use | Speedup | Install |
|-----------|-----|---------|---------|
| `pandas` | `polars` | 5-20x | `!pip install polars` |
| Slow NumPy loops | `numba` | 10-100x | Pre-installed in Colab |
| `scikit-learn` | `cuML` (GPU) | 10-50x | `!pip install cuml` |
| `pd.read_csv` | `polars.scan_csv` | Lazy, much faster | `!pip install polars` |

### Polars Example (Recommended Upgrade)
```python
import polars as pl

# Lazy evaluation — only processes what's needed
df = (
    pl.scan_csv('data.csv')
    .filter(pl.col('quantity') > 0)
    .with_columns(
        (pl.col('price') * pl.col('quantity')).alias('total')
    )
    .collect()  # Executes the full query at once
)
```

### Numba for Custom Math
```python
from numba import njit

@njit
def custom_calculation(prices, quantities, weights):
    n = len(prices)
    results = np.empty(n)
    for i in range(n):
        results[i] = prices[i] * quantities[i] * weights[i]
    return results

# Runs at near-C speed
results = custom_calculation(df['price'].values, df['quantity'].values, df['weight'].values)
```

---

## 6. Optimization-Specific Tips

If the project involves optimizing product choices (e.g., linear programming, portfolio selection):

### SciPy Optimize
```python
from scipy.optimize import minimize, linprog

# Vectorize the objective function — avoid Python loops inside it
def objective(x):
    return -np.dot(profits, x)  # Vectorized dot product

result = minimize(objective, x0, method='SLSQP', constraints=constraints)
```

### CVXPY for Convex Optimization
```python
!pip install cvxpy
import cvxpy as cp

x = cp.Variable(n_products)
objective = cp.Maximize(profits @ x)
constraints = [x >= 0, weights @ x <= capacity]
problem = cp.Problem(objective, constraints)
problem.solve()
```

### Scikit-learn at Scale
```python
# Use partial_fit for incremental learning
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for chunk in pd.read_csv('data.csv', chunksize=5000):
    X, y = chunk.drop('target', axis=1), chunk['target']
    model.partial_fit(X, y, classes=all_classes)
```

---

## 7. Google Colab-Specific Tips

### Enable GPU
- **Runtime > Change runtime type > GPU (T4)**
- Verify: `!nvidia-smi`

### Persist Data Between Sessions
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save intermediate results
df.to_parquet('/content/drive/MyDrive/processed_data.parquet')
```

### Cache Expensive Computations
```python
import pickle, os

cache_path = '/content/drive/MyDrive/cache/model.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = train_model(data)  # Expensive
    with open(cache_path, 'wb') as f:
        pickle.dump(model, f)
```

### Colab RAM Limits
- **Free tier**: ~12 GB RAM
- **Colab Pro**: ~25 GB RAM, longer runtimes, priority GPU
- If hitting limits, use chunked processing or Polars (more memory-efficient)

---

## 8. Quick Wins Checklist

- [ ] Profile code to find the actual slow part
- [ ] Replace all `for` loops with vectorized pandas/NumPy
- [ ] Convert string columns to `category` dtype
- [ ] Downcast numeric columns (float64 to float32)
- [ ] Switch CSV files to Parquet format
- [ ] Try Polars if pandas is the bottleneck
- [ ] Enable GPU runtime for ML training
- [ ] Mount Drive and cache intermediate results
- [ ] Only load needed columns with `usecols`

---

## Expected Performance at Different Scales

| Data Size | Pandas (unoptimized) | Pandas (optimized) | Polars |
|-----------|---------------------|-------------------|--------|
| 10K rows | Seconds | Sub-second | Sub-second |
| 100K rows | Minutes | Seconds | Seconds |
| 1M rows | May crash | Minutes | Seconds |
| 10M+ rows | Won't work | Challenging | Minutes |

---

*Generated 2026-02-08*
