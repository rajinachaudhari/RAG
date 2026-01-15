
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from semantic_search import semantic_search

###loading required files and models###
def main():
    chunks_df = pd.read_csv("data/processed/chunks_df.csv")
    embeddings = np.load("data/processed/embeddings.npy")
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    # User query
   #query = "How can I reset my password?"
    query = "will it fit a 2001 chevy suburban 1500?" 

    # Perform semantic search
    results = semantic_search(
        query=query,
        model=model,
        embeddings=embeddings,
        data=chunks_df,
        top_k=3
    )
#     # Truncate text ONLY for display
#     display_results = results.copy()
#     display_results["text"] = display_results["text"].apply(
#     lambda x: x[:20] + "..." if len(x) > 20 else x
# )
    # Display results
    print("\n Query:", query)
    print("\n Top matching chunks:\n")
    print(results.to_string(index=False)) #to print whole text 
    # print(display_results.to_string(index=False))  
    #to display truncated text only 
    #results:     
# domain                    text            score    
# customer_support use that. The cables... 0.691901    
# customer_support More it appears poss... 0.691627    
# customer_support cable either, so I'm... 0.664308 
   

if __name__ == "__main__":
    main()

# query = "Can you get the extra wires needed on amazon?"
# for q in queries:
#     print("\nQuery:", q)
#     display(semantic_search(q, model, embeddings, chunks_df,))
