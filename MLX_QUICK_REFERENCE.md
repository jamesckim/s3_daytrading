# MLX Training - Quick Reference

## The One Command You Need

```bash
# Just run this - uses your preferred defaults
python train_mlx.py
```

That's it! It automatically uses:
- Your curated tickers (james_tickers.json)
- 20 days of minute data
- 15 epochs of training
- Smart data pre-checking
- Optimized AVWAP computation

## Common Variations

```bash
# Quick test (5 epochs)
python train_mlx.py --epochs 5

# More data (30 days)
python train_mlx.py --days 30

# All 655 tickers
python train_mlx.py --tickers-file ../tickers/all_tickers.json

# Even simpler (if you chmod +x mlx)
./mlx --epochs 5
```

## Customize Your Defaults

Edit `mlx_config.json` to change your preferred settings:

```json
{
  "defaults": {
    "tickers_file": "../tickers/james_tickers.json",
    "days": 20,
    "epochs": 15,
    "sample_interval": 15
  }
}
```

## What Happens Behind the Scenes

1. **Always uses latest improvements** - Currently `train_mlx_with_precheck.py`
2. **Pre-checks data** - Won't waste time on processing if insufficient data
3. **Shows estimates** - Tells you how many samples before starting
4. **Smart defaults** - Your settings from mlx_config.json
5. **Override anything** - Command line args override defaults

## No More Remembering File Names!

Old way (confusing):
```bash
python train_indicator_transformer_mlx.py        # ❌ Old version
python train_mlx_intraday_final.py              # ❌ Which final?
python train_mlx_all_tickers.py                 # ❌ Specific version
python train_mlx_with_precheck.py               # ❌ Latest but hard to remember
```

New way (simple):
```bash
python train_mlx.py                              # ✅ Always right!
```

## Updates Are Automatic

When improvements are made:
1. Developer updates `LATEST_TRAINER` in `train_mlx.py`
2. You keep using `python train_mlx.py`
3. You automatically get the improvements!

No need to remember new file names ever again.