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

## 9. Dask — Parallel Processing Without Leaving Python

Dask scales pandas-like code across multiple cores. Good middle ground when pandas is too slow but you don't want to learn a new API.

```python
!pip install dask[dataframe]
import dask.dataframe as dd

# Reads in parallel, lazy by default
ddf = dd.read_csv('big_file.csv')

# Familiar pandas-like API
result = (
    ddf.groupby('category')['revenue']
    .sum()
    .compute()  # Triggers execution
)
```

### When to Use Dask vs Polars

| | Dask | Polars |
|---|------|--------|
| **API** | Nearly identical to pandas | Similar but different |
| **Best for** | Datasets larger than RAM | Fast single-machine processing |
| **Parallelism** | Multi-core + can scale to clusters | Multi-core, automatic |
| **Learning curve** | Minimal if you know pandas | Small but worth it |
| **Ecosystem** | Works with scikit-learn (`dask-ml`) | Growing, some gaps |

**Rule of thumb**: Try Polars first (faster, simpler). Use Dask if you need pandas compatibility or cluster scaling.

---

## 10. Feature Engineering at Scale

If training models on product data with many variables, feature engineering is often the bottleneck.

### Handle High-Cardinality Categoricals

```python
# Bad: One-hot encoding 10,000 product IDs = 10,000 new columns
# pd.get_dummies(df['product_id'])  # Will explode memory

# Good: Target encoding (replaces category with mean of target)
!pip install category_encoders
from category_encoders import TargetEncoder

encoder = TargetEncoder(cols=['product_id', 'supplier'])
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)
```

### Efficient Encoding Strategies

| Strategy | When to Use | Memory Impact |
|----------|-------------|---------------|
| **Target encoding** | High-cardinality (1000+ categories) | Low |
| **Frequency encoding** | When count matters (popular products) | Low |
| **One-hot encoding** | Low-cardinality only (< 20 categories) | High |
| **Hashing trick** | Very high cardinality, quick and dirty | Fixed |
| **Embeddings** | If using neural networks | Low |

```python
# Frequency encoding — simple and effective
freq = df['product_id'].value_counts(normalize=True)
df['product_freq'] = df['product_id'].map(freq)

# Hashing trick — fixed memory regardless of cardinality
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=128, input_type='string')
hashed = hasher.transform(df['product_id'].values.reshape(-1, 1))
```

### Batch Feature Generation

```python
# Generate features in chunks to avoid memory spikes
def generate_features(chunk):
    chunk['price_per_unit'] = chunk['price'] / chunk['quantity'].clip(lower=1)
    chunk['log_revenue'] = np.log1p(chunk['revenue'])
    chunk['price_bucket'] = pd.cut(chunk['price'], bins=10, labels=False)
    return chunk

chunks = [generate_features(c) for c in pd.read_csv('data.csv', chunksize=10_000)]
df = pd.concat(chunks)
```

---

## 11. Visualization at Scale

Plotting 10K+ points in Colab can freeze the browser. Use the right tool for the job.

### Matplotlib — Downsample First

```python
import matplotlib.pyplot as plt

# Don't plot all points — sample or aggregate
sample = df.sample(n=5000) if len(df) > 5000 else df
plt.scatter(sample['x'], sample['y'], alpha=0.3, s=5)
plt.show()
```

### Plotly — Interactive but Watch the Size

```python
!pip install plotly
import plotly.express as px

# Use WebGL renderer for large datasets
fig = px.scatter(
    df.sample(10_000),
    x='price', y='revenue',
    color='category',
    render_mode='webgl'  # Key for performance
)
fig.show()
```

### Datashader — Millions of Points, No Problem

```python
!pip install datashader holoviews bokeh
import datashader as ds
import datashader.transfer_functions as tf

canvas = ds.Canvas(plot_width=800, plot_height=600)
agg = canvas.points(df, 'x', 'y')
img = tf.shade(agg)
img
```

### Which Tool When

| Points | Tool | Notes |
|--------|------|-------|
| < 5K | Matplotlib / Seaborn | Fine as-is |
| 5K–50K | Plotly with `webgl` | Interactive, reasonable speed |
| 50K–1M | Datashader | Renders any size instantly |
| Any | Aggregated plots (histograms, box plots) | Always fast |

---

## 12. Common Pitfalls

Mistakes that silently kill performance:

| Pitfall | Why It's Slow | Fix |
|---------|---------------|-----|
| `df.apply(lambda x: ...)` on every row | Still a Python loop under the hood | Use vectorized operations |
| Copying DataFrames unnecessarily | `df2 = df` doesn't copy, but `df.copy()` does — know when you need it | Use `inplace=True` or chain operations |
| String operations in loops | Python strings are slow | Use `df['col'].str.method()` |
| Not using `.values` or `.to_numpy()` | Pandas Series has overhead vs raw arrays | Extract NumPy arrays for math-heavy code |
| Loading full dataset to filter it | Reads everything into memory first | Use `chunksize` or Polars `scan_csv` with `.filter()` |
| Repeated `.groupby()` calls | Each one scans the data | Combine into one groupby with `.agg()` |
| Ignoring index | Unindexed lookups are O(n) | `.set_index()` on frequently queried columns |

---

## Further Reading and Resources

### Official Documentation
- [Pandas: Enhancing Performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) — official optimization guide
- [Polars User Guide](https://docs.pola.rs/) — getting started with Polars
- [NumPy for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — vectorization fundamentals
- [Numba Documentation](https://numba.readthedocs.io/) — JIT compilation for Python
- [Dask Documentation](https://docs.dask.org/) — parallel computing in Python
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) — resource limits and tips

### Optimization and ML at Scale
- [Scikit-learn: Scaling Strategies](https://scikit-learn.org/stable/computing/scaling_strategies.html) — incremental learning, out-of-core
- [CVXPY Documentation](https://www.cvxpy.org/) — convex optimization in Python
- [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) — optimization algorithms
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders/) — encoding strategies for categorical data
- [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/) — GPU-accelerated DataFrames

### Visualization
- [Datashader Documentation](https://datashader.org/) — rendering billions of points
- [Plotly Python](https://plotly.com/python/) — interactive plotting with WebGL support

### Tutorials and Articles
- [Real Python: Fast, Flexible, Easy and Intuitive — Pandas](https://realpython.com/fast-flexible-pandas/) — practical pandas optimization
- [Towards Data Science: Speed Up Pandas](https://towardsdatascience.com/speed-up-your-pandas-processing/) — common speedup patterns
- [Jake VanderPlas: Why Python Is Slow](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/) — understanding the fundamentals

---

*Generated 2026-02-08 | Contributions welcome — open a PR or issue.*
