"""
ml_pipeline.py
==============
Sentiment Analysis using a Simple RNN (Keras / TensorFlow).

Architecture:
    Input text
        ↓
    Tokenizer  (word → integer index)
        ↓
    Embedding layer  (integer → dense vector, learned during training)
        ↓
    SimpleRNN layer  (reads sequence left-to-right, keeps hidden state)
        ↓
    Dense(1, sigmoid)  (outputs probability: 0=negative, 1=positive)

Dataset columns expected:
    ItemID  |  Sentiment (0/1)  |  SentimentText

Usage:
    python ml_pipeline.py --data dataset.csv

Output (saved to models/ folder):
    models/rnn_model.keras       ← trained RNN model
    models/tokenizer.pkl         ← fitted Keras Tokenizer
    models/config.pkl            ← max_len + vocab_size used during training
"""

import os
import re
import string
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import nltk

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF C++ info logs
import tensorflow as tf
from tensorflow import keras  # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Embedding, SimpleRNN, Dense, Dropout
from keras.preprocessing.text import Tokenizer # type: ignore
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

# ── NLTK ──────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
VOCAB_SIZE   = 20_000   # top N words to keep in vocabulary
MAX_LEN      = 50       # pad/truncate every tweet to this many tokens
EMBED_DIM    = 64       # size of each word embedding vector
RNN_UNITS    = 64       # number of SimpleRNN hidden units
DROPOUT_RATE = 0.3      # dropout to prevent overfitting
BATCH_SIZE   = 64
EPOCHS       = 10       # EarlyStopping will stop before this if needed

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "rnn_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH    = os.path.join(MODEL_DIR, "config.pkl")

LABEL_MAP    = {0: "negative", 1: "positive"}
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Reads CSV with columns: ItemID, Sentiment (0/1), SentimentText
    Returns cleaned DataFrame with columns: review, sentiment (0/1 as int)
    """
    log.info(f"Loading dataset: {csv_path}")

    df = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            log.info(f"  Encoding : {enc}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Cannot read {csv_path}. Try saving as UTF-8.")

    df.columns = [c.strip().lower() for c in df.columns]

    # Rename text column
    if "sentimenttext" in df.columns:
        df = df.rename(columns={"sentimenttext": "review"})
    elif "text" in df.columns:
        df = df.rename(columns={"text": "review"})
    else:
        raise ValueError(f"Need 'SentimentText' column. Found: {list(df.columns)}")

    if "sentiment" not in df.columns:
        raise ValueError(f"Need 'Sentiment' column. Found: {list(df.columns)}")

    # Keep sentiment as 0/1 integer (RNN uses sigmoid output → binary crossentropy)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").astype("Int64")

    rows_before = len(df)
    df = df[["review", "sentiment"]].dropna()
    df = df[df["sentiment"].isin([0, 1])]
    df = df.drop_duplicates(subset=["review"])
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 3]

    log.info(f"  Rows loaded  : {rows_before}")
    log.info(f"  After cleanup: {len(df)}")
    log.info(f"  Label counts :\n{df['sentiment'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Clean Text
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans a tweet for RNN input.

    Note: for RNNs we keep MORE words than TF-IDF models because
    the RNN learns context (word order matters). We still remove
    noise like @mentions, URLs, and punctuation.
    """
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)              # remove @mentions
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = re.sub(r"#(\w+)", r"\1", text)          # #hashtag → hashtag
    text = re.sub(r"<[^>]+>", "", text)            # remove HTML
    text = text.translate(
        str.maketrans("", "", string.punctuation)  # remove punctuation
    )
    text = re.sub(r"\d+", "", text)                # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning tweet text...")
    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.strip().astype(bool)]
    log.info(f"Rows after cleaning: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Tokenize + Pad sequences
# ─────────────────────────────────────────────────────────────────────────────
def build_tokenizer(texts: np.ndarray) -> Tokenizer:
    """
    Fits a Keras Tokenizer on the training corpus.

    What this does:
      "i love this movie"  →  [4, 12, 6, 89]
    Each word gets a unique integer index ranked by frequency.
    """
    log.info(f"Building vocabulary (max {VOCAB_SIZE} words)...")
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>",    # out-of-vocabulary token for unseen words
    )
    tokenizer.fit_on_texts(texts)
    vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    log.info(f"  Vocabulary size : {vocab_size}")
    return tokenizer


def texts_to_sequences(tokenizer: Tokenizer, texts: np.ndarray) -> np.ndarray:
    """
    Converts text list → padded integer matrix of shape (N, MAX_LEN).

    Why padding?
      RNN needs fixed-length input. Short tweets get zeros appended (post-pad).
      Long tweets get truncated from the end (post-trunc).

    Example (MAX_LEN=6):
      "love this"  →  [12, 6, 0, 0, 0, 0]
      "i really really love this movie now"  →  [4, 99, 99, 12, 6, 89]
    """
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Build RNN Model
# ─────────────────────────────────────────────────────────────────────────────
def build_rnn(vocab_size: int) -> keras.Model:
    """
    Simple RNN architecture:

        [word indices]   shape: (batch, MAX_LEN)
              ↓
        Embedding        shape: (batch, MAX_LEN, EMBED_DIM)
          Learns a dense vector for each word.
          Similar words (good, great, awesome) cluster together.
              ↓
        SimpleRNN        shape: (batch, RNN_UNITS)
          Reads the sequence left-to-right.
          At each step: h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
          Only the LAST hidden state is passed forward.
              ↓
        Dropout          randomly zeros units during training (regularisation)
              ↓
        Dense(1, sigmoid)   shape: (batch, 1)
          Output ∈ [0, 1]
          > 0.5  →  positive
          ≤ 0.5  →  negative
    """
    actual_vocab = min(vocab_size, VOCAB_SIZE) + 1  # +1 for padding index 0

    model = Sequential([
        Embedding(
            input_dim    = actual_vocab,
            output_dim   = EMBED_DIM,
            input_length = MAX_LEN,
            name         = "embedding",
        ),
        SimpleRNN(
            units            = RNN_UNITS,
            activation       = "tanh",
            return_sequences = False,   # only return the final hidden state
            name             = "simple_rnn",
        ),
        Dropout(DROPOUT_RATE, name="dropout"),
        Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer = "adam",
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Train
# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    """
    Full training flow:
      1. Split data
      2. Fit tokenizer on training text
      3. Convert text → padded sequences
      4. Build RNN
      5. Train with EarlyStopping
      6. Evaluate on test set
    Returns: (model, tokenizer)
    """
    X = df["clean_review"].values
    y = df["sentiment"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    log.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Tokenize ──────────────────────────────────────────────────────────
    tokenizer = build_tokenizer(X_train)       # fit ONLY on training data
    X_train_seq = texts_to_sequences(tokenizer, X_train)
    X_test_seq  = texts_to_sequences(tokenizer, X_test)

    vocab_size = len(tokenizer.word_index) + 1
    log.info(f"Sequence shape — train: {X_train_seq.shape}  test: {X_test_seq.shape}")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_rnn(vocab_size)
    model.summary(print_fn=log.info)

    # ── Callbacks ─────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor   = "val_loss",
            patience  = 2,            # stop if val_loss doesn't improve for 2 epochs
            restore_best_weights = True,
            verbose   = 1,
        ),
        ModelCheckpoint(
            filepath       = MODEL_PATH,
            save_best_only = True,
            monitor        = "val_accuracy",
            verbose        = 1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    log.info(f"Training RNN — max {EPOCHS} epochs, batch {BATCH_SIZE}...")
    history = model.fit(
        X_train_seq, y_train,
        validation_split = 0.1,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
    print("\n" + "=" * 50)
    print(f"  Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Test Loss     : {loss:.4f}")
    print("=" * 50)

    # Detailed report
    y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index   = ["actual_neg", "actual_pos"],
        columns = ["pred_neg",   "pred_pos"],
    )
    print(cm_df.to_string())
    print("=" * 50 + "\n")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Save model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model: keras.Model, tokenizer: Tokenizer):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save RNN model (native Keras format)
    model.save(MODEL_PATH)
    log.info(f"RNN model saved   →  {MODEL_PATH}")

    # Save tokenizer (needed to convert new text → sequences at inference time)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    log.info(f"Tokenizer saved   →  {TOKENIZER_PATH}")

    # Save config (MAX_LEN must match what was used during training)
    config = {"max_len": MAX_LEN, "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM}
    joblib.dump(config, CONFIG_PATH)
    log.info(f"Config saved      →  {CONFIG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  Load artifacts  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts() -> tuple:
    """
    Returns (model, tokenizer, config)
    Called once at API startup in main.py.
    """
    for path in [MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Train first:  python ml_pipeline.py --data dataset.csv"
            )
    model     = keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    config    = joblib.load(CONFIG_PATH)
    log.info("RNN model + tokenizer loaded successfully")
    return model, tokenizer, config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  Predict  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment(model: keras.Model, tokenizer: Tokenizer,
                      config: dict, text: str) -> dict:
    """
    End-to-end prediction for a single raw text string.

    Flow:
        raw text  →  clean  →  tokenize  →  pad  →  RNN  →  sigmoid  →  label
    """
    max_len = config["max_len"]

    cleaned  = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")

    prob_positive = float(model.predict(padded, verbose=0)[0][0])
    prob_negative = round(1.0 - prob_positive, 4)
    prob_positive = round(prob_positive, 4)

    sentiment  = "positive" if prob_positive >= 0.5 else "negative"
    confidence = max(prob_positive, prob_negative)

    return {
        "sentiment":     sentiment,
        "confidence":    round(confidence, 4),
        "probabilities": {
            "negative": prob_negative,
            "positive": prob_positive,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RNN sentiment analysis model"
    )
    parser.add_argument(
        "--data",
        type    = str,
        default = "dataset.csv",
        help    = "Path to CSV with columns: ItemID, Sentiment, SentimentText",
    )
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  Twitter Sentiment — Simple RNN Pipeline")
    log.info("=" * 55)
    log.info(f"  TensorFlow version : {tf.__version__}")
    log.info(f"  GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

    df             = load_data(args.data)
    df             = preprocess(df)
    model, tokenizer = train(df)
    save_artifacts(model, tokenizer)

    # ── Sanity check: tweets from your screenshot ─────────────────────────
    model_loaded, tok_loaded, cfg_loaded = load_artifacts()

    samples = [
        ("is so sad for my APL friend............",  "negative"),
        ("I missed the New Moon trailer...",          "negative"),
        ("omg its already 7:30 :O",                  "positive"),
        ("Juuuuuuussst Chillin!!",                    "positive"),
        ("i think mi bf is cheating on me!!! T_T",   "negative"),
        ("thanks to all the haters up in my face",   "positive"),
        ("handed in my uniform today . i miss you",  "positive"),
    ]

    print("\n── Sanity Check ─────────────────────────────────────")
    print(f"{'Tweet':<46}  {'Actual':<10} {'Pred':<10} {'Conf':>6}")
    print("-" * 76)
    for tweet, actual in samples:
        r    = predict_sentiment(model_loaded, tok_loaded, cfg_loaded, tweet)
        tick = "OK" if r["sentiment"] == actual else "!!"
        print(
            f"{tweet[:45]:<46}  {actual:<10} {r['sentiment']:<10} "
            f"{r['confidence']*100:5.1f}%  {tick}"
        )

    log.info("\nDone. Start API:  uvicorn main:app --reload")