# Research Resources

> A collection of practical guides and references for data science and software engineering.

---

## Guides

| Guide | Description |
|-------|-------------|
| [Python/Colab Optimization](./Python-Colab-Optimization-Guide.md) | Scale Python data processing from 10K to 10M+ rows in Google Colab |

---

## Python/Colab Optimization — At a Glance

Struggling with slow Python code in Google Colab? Here's the quick version:

### The Big 3 Fixes

**1. Vectorize** — Replace `for` loops with pandas/NumPy operations (10-100x faster)

```python
# Instead of looping row by row...
df['total'] = df['price'] * df['quantity']
```

**2. Use Parquet** — Drop CSV, use Parquet for faster I/O and smaller files

```python
df.to_parquet('data.parquet')
df = pd.read_parquet('data.parquet')
```

**3. Try Polars** — Drop-in pandas alternative, 5-20x faster

```python
import polars as pl
df = pl.scan_csv('data.csv').filter(pl.col('qty') > 0).collect()
```

### Performance Expectations

```
         Pandas (raw)    Pandas (tuned)    Polars
10K      seconds         sub-second        sub-second
100K     minutes         seconds           seconds
1M       may crash       minutes           seconds
10M+     won't work      challenging       minutes
```

**[Read the full guide →](./Python-Colab-Optimization-Guide.md)**

---

## Contributing

Have a topic you'd like covered? Open an issue or PR.

## License

MIT
