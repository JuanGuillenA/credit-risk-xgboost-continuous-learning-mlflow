import os
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "duration", "credit_amount", "age",
    "checking_status", "employment", "savings_status", "purpose"
]
TARGET = "target"

CLEAN_PATH = "artifacts/dataset/german_credit_clean.csv"

OUT_FIXED_TEST = "artifacts/dataset/fixed_test.csv"
OUT_TRAIN_POOL = "artifacts/dataset/train_pool.csv"
OUT_BASE50 = "artifacts/dataset/initial_base_50.csv"
OUT_FUTURE_POOL = "artifacts/dataset/future_pool.csv"

BASE50_SIZE = 50
TEST_SIZE = 0.25
SEED_SPLIT = 42
SEED_BASE50 = 101
SEED_FUTURE = 202

def ensure_cols(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] faltan columnas: {missing}. Tengo: {list(df.columns)}")

def main():
    os.makedirs("artifacts/dataset", exist_ok=True)

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"No existe {CLEAN_PATH}")

    df = pd.read_csv(CLEAN_PATH)
    ensure_cols(df, FEATURES + [TARGET], "german_credit_clean")
    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Split fijo: train_pool y fixed_test
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED_SPLIT, stratify=y
    )

    fixed_test = X_test.copy()
    fixed_test[TARGET] = y_test.values
    fixed_test.to_csv(OUT_FIXED_TEST, index=False)

    train_pool = X_train.copy()
    train_pool[TARGET] = y_train.values
    train_pool.to_csv(OUT_TRAIN_POOL, index=False)

    # base50 sale SOLO del train_pool
    if len(train_pool) < BASE50_SIZE:
        raise ValueError(f"train_pool tiene {len(train_pool)} filas y base50 pide {BASE50_SIZE}")

    base50 = train_pool.sample(BASE50_SIZE, random_state=SEED_BASE50)
    base50.to_csv(OUT_BASE50, index=False)

    # future_pool = resto del train_pool
    remaining = train_pool.drop(base50.index)
    future_pool = remaining.sample(len(remaining), random_state=SEED_FUTURE)
    future_pool.to_csv(OUT_FUTURE_POOL, index=False)

    print("Opción A datasets creados:")
    print(" -", OUT_FIXED_TEST, "rows:", len(fixed_test))
    print(" -", OUT_TRAIN_POOL, "rows:", len(train_pool))
    print(" -", OUT_BASE50, "rows:", len(base50))
    print(" -", OUT_FUTURE_POOL, "rows:", len(future_pool))

if __name__ == "__main__":
    main()
