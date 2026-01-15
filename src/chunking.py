from typing import List

###creating chunk function###

def chunk_text( text: str, chunk_size: int = 40, overlap: int = 10) -> List[str]:
    words = text.split()
    chunks = []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


###to run below pipeline first load documents_sampled_cleaned.txt file into 1 variable###
file_path = "data/processed/documents_sampled_clean.txt"

with open(file_path, "r", encoding="utf-8") as f:
    document_text = f.read()
    
    
###chunk pipeline function###
 
import pandas as pd

chunked_docs = []

domain = "customer_support"


chunks = chunk_text(document_text)


for chunk in chunks:
    row = {
        "domain": domain,
        "text": chunk
    }
    chunked_docs.append(row)


chunks_df = pd.DataFrame(chunked_docs)

print(chunks_df.head())

###now inspec the chunk made###

print("Total chunks:", len(chunks))    #print total number of chunks

###Save DataFrame for embedding.py###
chunks_df.to_csv("data/processed/chunks_df.csv", index=False)
print(chunks_df.head())
