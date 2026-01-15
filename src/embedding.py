import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


###Load chunked DataFrame###
chunks_df = pd.read_csv("data/processed/chunks_df.csv")

###selecting models###
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

###embedding chunks###
# texts = chunks_df["text"].tolist()

# embeddings = model.encode(texts, convert_to_numpy=True)

embeddings = model.encode(
    chunks_df["text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)
chunks_df["embeddings"] = list(embeddings)
chunks_df.head()

###save values of embedding
np.save("data/processed/embeddings.npy", embeddings)


print("Embeddings shape:", embeddings.shape)
