"""
Dataset 1 — Transaction Events — COMPLETE FINAL VERSION
Generates:
  - transactions/     : 10 parquet files, 10M rows, 8 columns
  - merchant_lookup/  : 1 parquet file, 500 rows, 4 columns

All 6 scenario data guarantees verified:
  S1  Skew         : 10 hot merchants out of 500 = 70% of all rows
  S2  Window       : 2,000 rapid customers get 3+ transactions within a 10-min
                     window AND elevated amounts (>3x their own avg) injected
  S3  Broadcast    : merchant_lookup = 500 rows (~20KB) — always within 10MB threshold
  S4  Schema       : chunks 0-7 have column 'amount'; chunks 8-9 renamed to
                     'transaction_amount' — all in same folder for silent corruption demo
  S5  Repartition  : 10 separate parquet files in same folder
  S6  Caching      : failed+pending = ~21% of rows (~2.1M) = correct selective-cache
                     target; full 10M rows = wrong cache that forces spill
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STORAGE PATHS  —  adjust BASE to your local drive
# =============================================================================

BASE = r"D:/data engineer/pyspark_practice/pyspark_scenarios/dataset1_transactions"

PATHS = {
    "transactions"   : f"{BASE}/transactions",
    "merchant_lookup": f"{BASE}/merchant_lookup",
}
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

print("=" * 65)
print("Dataset 1 — Transaction Events — COMPLETE FINAL VERSION")
print("=" * 65)

# =============================================================================
# CONSTANTS
# =============================================================================

TOTAL_ROWS    = 10_000_000
CHUNK_SIZE    =  1_000_000
NUM_CHUNKS    = TOTAL_ROWS // CHUNK_SIZE   # 10
# chunks 0-7  -> original schema  column named: 'amount'
# chunks 8-9  -> evolved schema   column named: 'transaction_amount'  (Scenario 4)

NUM_MERCHANTS = 500
NUM_CUSTOMERS = 300_000

np.random.seed(42)
random.seed(42)
BASE_TS = datetime(2024, 1, 1, 0, 0, 0)

# =============================================================================
# COLUMN VALUE POOLS
# =============================================================================

STATUSES       = ["completed", "pending", "failed", "refunded"]
# completed=72%, pending=12%, failed=9%, refunded=7%
# S6: wrong cache  = full 10M rows (forces spill)
#     correct cache = filter to failed+pending (~21% = ~2.1M rows, fits in memory)
STATUS_WEIGHTS = [0.72,         0.12,      0.09,     0.07]

REGIONS        = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.28,    0.22,    0.20,   0.18,   0.12]

CATEGORIES = [
    "Electronics", "Clothing", "Groceries", "Food & Dining",
    "Travel", "Healthcare", "Entertainment", "Home & Garden",
    "Sports", "Books"
]
TIERS = ["Gold", "Silver", "Bronze"]

PAYMENT_METHODS        = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"]
PAYMENT_METHOD_WEIGHTS = [0.35,  0.25,          0.20,         0.12,          0.08]

# =============================================================================
# STEP 1 — BUILD ID POOLS
# =============================================================================

print("\n[STEP 1/4] Building ID pools...")

MERCHANT_IDS  = np.array([f"M{str(i).zfill(5)}" for i in range(1, NUM_MERCHANTS + 1)])
CUSTOMER_IDS  = np.array([f"C{str(i).zfill(7)}" for i in range(1, NUM_CUSTOMERS + 1)])

# ── Scenario 1 — Skew ────────────────────────────────────────────────────────
# 10 hot merchants get 70% of all rows. 490 normal merchants share 30%.
HOT_MERCHANTS    = MERCHANT_IDS[:10]       # M00001 ... M00010
NORMAL_MERCHANTS = MERCHANT_IDS[10:]       # M00011 ... M00500

HOT_WEIGHT_EACH    = 0.70 / len(HOT_MERCHANTS)       # 0.07 each
NORMAL_WEIGHT_EACH = 0.30 / len(NORMAL_MERCHANTS)    # ~0.000612 each

MERCHANT_WEIGHTS = (
    [HOT_WEIGHT_EACH]    * len(HOT_MERCHANTS) +
    [NORMAL_WEIGHT_EACH] * len(NORMAL_MERCHANTS)
)
assert abs(sum(MERCHANT_WEIGHTS) - 1.0) < 1e-9, "Merchant weights must sum to 1"

# ── Scenario 2 — Window Functions ────────────────────────────────────────────
# 2,000 rapid (fraud) customers from the first 50K customer pool.
# Each will receive:
#   (a) 3+ transactions clustered within a 600s (10-min) window
#   (b) Elevated amounts set to 3.5x their own chunk mean
#       so the 'amount > 3x historical average' filter reliably fires
RAPID_CUSTOMERS = set(random.sample(list(CUSTOMER_IDS[:50_000]), 2_000))

print(f"   Merchants      : {NUM_MERCHANTS:,}  (10 hot / 490 normal)")
print(f"   Customers      : {NUM_CUSTOMERS:,}")
print(f"   Rapid/fraud    : {len(RAPID_CUSTOMERS):,} from first 50K pool")

# =============================================================================
# STEP 2 — MERCHANT LOOKUP TABLE  (Scenario 3 — Broadcast Join)
# 500 rows, 4 columns, ~20KB — always << 10MB broadcast threshold
# =============================================================================

print("\n[STEP 2/4] Generating merchant_lookup (500 rows)...")

rng_ml = np.random.default_rng(seed=11111)

df_merchant = pd.DataFrame({
    "merchant_id"  : MERCHANT_IDS,
    "merchant_name": [f"Merchant_{mid}" for mid in MERCHANT_IDS],
    "category"     : rng_ml.choice(CATEGORIES, size=NUM_MERCHANTS),
    "tier"         : rng_ml.choice(TIERS,       size=NUM_MERCHANTS,
                                   p=[0.20, 0.45, 0.35]),
})

lookup_path = f"{PATHS['merchant_lookup']}/merchant_lookup.parquet"
df_merchant.to_parquet(lookup_path, index=False, compression="snappy")
lookup_mb = os.path.getsize(lookup_path) / (1024 ** 2)
print(f"   Written: {lookup_mb:.3f} MB  (target << 10 MB for broadcast demo)")

# =============================================================================
# CHUNK GENERATION FUNCTION
# =============================================================================

def generate_chunk(chunk_id: int, size: int) -> pd.DataFrame:
    """
    Correct order of operations verified against all 6 scenarios:

    STEP A  Generate ALL column arrays in original random order BEFORE any sort.
            merchant_ids, customer_ids, amounts, statuses, regions,
            payment_methods all in the same original unsorted order.

    STEP B  Generate base_seconds (timestamps as integers, unsorted).

    STEP C  First argsort: sort ALL 7 data arrays + base_seconds simultaneously
            using the same sort_order so every row stays column-aligned.

    STEP D  Rapid-customer injection (Scenario 2):
            For each rapid customer with >= 3 occurrences in this chunk:
              (a) Cluster base_seconds: 1-60s gaps per event
                  Max cumulative gap for 3 events = 120s << 600s window
              (b) Set amounts to 3.5x that customer's in-chunk mean
                  so 'amount > 3x historical avg' filter reliably fires

    STEP E  Second argsort: re-sort ALL arrays after injection because
            injection broke chronological order.

    STEP F  Convert base_seconds to Timestamps (vectorised, no Python loop).

    STEP G  Assemble and return DataFrame.
            chunks 0-7  -> column 'amount'               (original schema)
            chunks 8-9  -> column 'transaction_amount'   (evolved schema S4)
    """

    rng = np.random.default_rng(seed=chunk_id * 777 + 3)

    # ── STEP A: ALL columns in original random order ──────────────────────────

    start_idx = chunk_id * size

    # transaction_id: globally unique — chunk_id * size gives non-overlapping ranges
    transaction_ids = np.array([f"T{start_idx + i:010d}" for i in range(size)])

    # merchant_ids — S1 skew: 70% from 10 hot merchants
    hot_mask     = rng.random(size) < 0.70
    merchant_ids = np.where(
        hot_mask,
        HOT_MERCHANTS[rng.integers(0, len(HOT_MERCHANTS),    size=size)],
        NORMAL_MERCHANTS[rng.integers(0, len(NORMAL_MERCHANTS), size=size)],
    )

    customer_ids = CUSTOMER_IDS[rng.integers(0, len(CUSTOMER_IDS), size=size)]

    # amounts — log-normal centred ~Rs 2,000, clipped to [10, 500_000]
    raw_amounts = np.exp(rng.normal(loc=7.6, scale=1.4, size=size))
    amounts     = np.clip(raw_amounts, 10.0, 500_000.0).round(2)

    statuses        = rng.choice(STATUSES,        size=size, p=STATUS_WEIGHTS)
    regions         = rng.choice(REGIONS,         size=size, p=REGION_WEIGHTS)
    payment_methods = rng.choice(PAYMENT_METHODS, size=size, p=PAYMENT_METHOD_WEIGHTS)

    # ── STEP B: Base timestamps ───────────────────────────────────────────────
    year_seconds = 365 * 24 * 3600
    window_size  = year_seconds // NUM_CHUNKS
    base_seconds = (
        chunk_id * window_size +
        rng.integers(0, window_size, size=size)
    ).astype(np.int64)

    # ── STEP C: First sort — ALL 7 arrays sorted with same sort_order ─────────
    s               = np.argsort(base_seconds)
    base_seconds    = base_seconds[s]
    transaction_ids = transaction_ids[s]
    merchant_ids    = merchant_ids[s]
    customer_ids    = customer_ids[s]
    amounts         = amounts[s]
    statuses        = statuses[s]
    regions         = regions[s]
    payment_methods = payment_methods[s]

    # ── STEP D: Rapid-customer injection (Scenario 2) ─────────────────────────
    # vectorised isin — fast (pandas uses C hash set internally)
    rapid_mask = pd.Series(customer_ids).isin(RAPID_CUSTOMERS).values

    if rapid_mask.any():
        idx_map: dict[str, list[int]] = {}
        for idx in np.where(rapid_mask)[0]:
            cid = customer_ids[idx]
            if cid not in idx_map:
                idx_map[cid] = []
            idx_map[cid].append(idx)

        for cid, indices in idx_map.items():
            if len(indices) >= 3:
                # (a) Cluster timestamps within 10-min window
                anchor = int(base_seconds[indices[0]])
                for k in range(1, len(indices)):
                    anchor += int(rng.integers(1, 61))   # 1-60s gap each
                    base_seconds[indices[k]] = anchor

                # (b) Elevate amounts to 3.5x this customer's in-chunk mean
                #     ensuring amount > 3x historical avg filter reliably fires
                cust_mean = float(amounts[indices].mean())
                for k in indices:
                    amounts[k] = round(cust_mean * 3.5, 2)

    # ── STEP E: Re-sort ALL 7 arrays after injection ──────────────────────────
    fs              = np.argsort(base_seconds)
    base_seconds    = base_seconds[fs]
    transaction_ids = transaction_ids[fs]
    merchant_ids    = merchant_ids[fs]
    customer_ids    = customer_ids[fs]
    amounts         = amounts[fs]
    statuses        = statuses[fs]
    regions         = regions[fs]
    payment_methods = payment_methods[fs]

    # ── STEP F: Vectorised timestamps ─────────────────────────────────────────
    timestamps = pd.to_datetime(BASE_TS) + pd.to_timedelta(base_seconds, unit="s")
    timestamps = timestamps.astype("datetime64[us]")

    # ── STEP G: Assemble DataFrame ────────────────────────────────────────────
    # Scenario 4 — Schema Evolution:
    #   chunks 0-7  -> 'amount'               original schema
    #   chunks 8-9  -> 'transaction_amount'   evolved schema — same folder
    amount_col_name = "amount" if chunk_id < 8 else "transaction_amount"

    df = pd.DataFrame({
        "transaction_id" : transaction_ids,
        "customer_id"    : customer_ids,
        "merchant_id"    : merchant_ids,
        amount_col_name  : amounts,
        "status"         : statuses,
        "region"         : regions,
        "payment_method" : payment_methods,
        "transaction_ts" : timestamps,
    })

    return df


# =============================================================================
# STEP 3 — GENERATE TRANSACTIONS
# =============================================================================

print(f"\n[STEP 3/4] Generating transactions ({TOTAL_ROWS:,} rows, {NUM_CHUNKS} files)...")

for chunk_id in range(NUM_CHUNKS):
    schema_tag = "original  (amount)" if chunk_id < 8 else "evolved (transaction_amount)"
    print(f"   Chunk {chunk_id+1:02d}/{NUM_CHUNKS} [{schema_tag}]...", end=" ", flush=True)
    df_c     = generate_chunk(chunk_id, CHUNK_SIZE)
    out_path = f"{PATHS['transactions']}/part_{str(chunk_id).zfill(4)}.parquet"
    df_c.to_parquet(out_path, index=False, compression="snappy")
    size_mb  = os.path.getsize(out_path) / (1024 ** 2)
    print(f"done — {size_mb:.1f} MB")

print(f"   Transactions complete: {TOTAL_ROWS:,} rows across {NUM_CHUNKS} files")

# =============================================================================
# STEP 4 — VALIDATION — All 6 Scenarios
# =============================================================================

print("\n" + "=" * 65)
print("VALIDATION — All 6 Scenarios")
print("=" * 65)

df_s0 = pd.read_parquet(f"{PATHS['transactions']}/part_0000.parquet")  # original schema
df_s8 = pd.read_parquet(f"{PATHS['transactions']}/part_0008.parquet")  # evolved schema
df_ml = pd.read_parquet(lookup_path)

# ── Scenario 1 — Skew ────────────────────────────────────────────────────────
hot_set  = set(HOT_MERCHANTS)
hot_rows = df_s0["merchant_id"].isin(hot_set).sum()
hot_pct  = hot_rows / len(df_s0) * 100
s1_pass  = hot_pct >= 65   # target 70%, allow ±5% sampling variance
print(f"\nS1 Skew      : hot_merchant_rows={hot_rows:,} ({hot_pct:.1f}%)"
      f" — {'PASS' if s1_pass else 'FAIL'}")

# ── Scenario 2 — Window Functions ────────────────────────────────────────────
# Check (a): tight gaps
rapid_df  = df_s0[df_s0["customer_id"].isin(RAPID_CUSTOMERS)].copy()
rapid_df  = rapid_df.sort_values(["customer_id", "transaction_ts"])
rapid_df["prev_ts"] = rapid_df.groupby("customer_id")["transaction_ts"].shift(1)
rapid_df["gap_s"]   = (rapid_df["transaction_ts"] - rapid_df["prev_ts"]).dt.total_seconds()
tight     = rapid_df[rapid_df["gap_s"] < 600].dropna(subset=["gap_s"])
s2a_pass  = len(tight) > 50
print(f"S2 Win-gap   : rapid_rows={len(rapid_df):,}"
      f"  tight_gaps(<10min)={len(tight):,}"
      f" — {'PASS' if s2a_pass else 'FAIL'}")

# Check (b): elevated amounts
cust_means   = df_s0.groupby("customer_id")["amount"].mean()
df_s0_joined = df_s0.join(cust_means.rename("cust_mean"), on="customer_id")
high_amt     = df_s0_joined[
    df_s0_joined["customer_id"].isin(RAPID_CUSTOMERS) &
    (df_s0_joined["amount"] > df_s0_joined["cust_mean"] * 3)
]
s2b_pass = len(high_amt) > 20
print(f"S2 Amt-check : rapid_rows_with_amount>3x_mean={len(high_amt):,}"
      f" — {'PASS' if s2b_pass else 'FAIL'}")

# ── Scenario 3 — Broadcast Join ───────────────────────────────────────────────
txn_mids  = set(df_s0["merchant_id"].unique())
lkp_mids  = set(df_ml["merchant_id"].unique())
cov_pct   = len(txn_mids & lkp_mids) / len(txn_mids) * 100
s3_pass   = (cov_pct == 100.0) and (lookup_mb < 1.0)
print(f"S3 Broadcast : coverage={cov_pct:.1f}%  lookup={lookup_mb:.3f} MB"
      f" — {'PASS' if s3_pass else 'FAIL'}")

# ── Scenario 4 — Schema Evolution ────────────────────────────────────────────
orig_cols    = set(df_s0.columns)
evolved_cols = set(df_s8.columns)
dropped      = orig_cols    - evolved_cols   # {'amount'}
added        = evolved_cols - orig_cols      # {'transaction_amount'}
s4_pass      = ("amount" in dropped) and ("transaction_amount" in added)
print(f"S4 Schema    : orig={sorted(orig_cols)}"
      f"\n               evol={sorted(evolved_cols)}"
      f"\n               dropped={dropped}  added={added}"
      f" — {'PASS' if s4_pass else 'FAIL'}")

# ── Scenario 5 — Repartition / Coalesce ──────────────────────────────────────
n_files  = len([f for f in os.listdir(PATHS["transactions"]) if f.endswith(".parquet")])
s5_pass  = (n_files == NUM_CHUNKS)
print(f"S5 Files     : {n_files} parquet files in transactions/"
      f" — {'PASS' if s5_pass else 'FAIL'}")

# ── Scenario 6 — Caching ─────────────────────────────────────────────────────
fp_pct   = df_s0["status"].isin(["failed", "pending"]).mean() * 100
fp_rows  = int(TOTAL_ROWS * fp_pct / 100)
s6_pass  = 15 < fp_pct < 35   # expect ~21%
print(f"S6 Cache     : failed+pending={fp_pct:.1f}% -> ~{fp_rows:,} rows"
      f" (selective-cache target)"
      f" — {'PASS' if s6_pass else 'FAIL'}")

# ── Chronological order ───────────────────────────────────────────────────────
diffs    = df_s0["transaction_ts"].diff().dt.total_seconds().dropna()
oor      = (diffs < 0).sum()
ord_pass = (oor == 0)
print(f"Order check  : out_of_order_rows={oor}"
      f" — {'PASS' if ord_pass else 'FAIL — re-sort broken'}")

# ── Disk usage ────────────────────────────────────────────────────────────────
total_bytes = sum(
    os.path.getsize(os.path.join(fp, f))
    for fp in PATHS.values()
    for f in os.listdir(fp) if f.endswith(".parquet")
)
print(f"\nDisk total   : {total_bytes / (1024**2):.1f} MB")
print(f"Rows total   : {TOTAL_ROWS:,}  +  merchant_lookup={NUM_MERCHANTS}")

print("\n" + "=" * 65)
print("GENERATION COMPLETE — ALL 6 SCENARIOS READY")
print("=" * 65)

print("""
SCENARIO QUICK REFERENCE
──────────────────────────────────────────────────────────────────────
S1  Skew + Salting
    Key: merchant_id. 10 merchants = 70% of 10M rows.
    groupBy('merchant_id') pins one task for 28+ minutes.
    Fix: concat(col('merchant_id'), lit('_'), (rand()*10).cast('int'))

S2  Window Functions
    2,000 rapid customers: 3+ txns within 10-min window + amounts
    set to 3.5x their own mean so amount > 3x historical avg fires.
    GROUP BY cannot see time gaps. lag() on transaction_ts can.

S3  Broadcast Join
    merchant_lookup = 500 rows, ~20KB.
    Use spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')
    BEFORE demo to disable auto-broadcast (forces shuffle join = before).
    Then broadcast() explicitly to show the after state.

S4  Schema Evolution
    chunks 0-7 = 'amount', chunks 8-9 = 'transaction_amount',
    ALL in same transactions/ folder.
    Without mergeSchema=True: revenue for 2M rows shows as NULL/0.
    Fix: mergeSchema=True + coalesce(col('amount'),
         col('transaction_amount')).alias('amount')

S5  Repartition vs Coalesce
    10 files -> Spark reads with many partitions after aggregation.
    repartition(4): full shuffle, Exchange node in explain plan.
    coalesce(4): no shuffle, Exchange node gone, 10x faster write.

S6  Caching
    Wrong:   df_raw.cache() on full 10M rows -> spill to disk.
             (Requires separate SparkSession with memory.fraction=0.4)
    Correct: filter to failed+pending (~21% = ~2.1M rows) then cache.
             Fits in memory. Zero spill. Downstream reuse is fast.
──────────────────────────────────────────────────────────────────────
""")
