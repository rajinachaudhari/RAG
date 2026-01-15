import pandas as pd
import os

INPUT_FILE = "data/raw/multi_answers.csv"
OUTPUT_FILE = "data/processed/documents.txt"

def csv_to_documents():
    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for chunk in pd.read_csv(INPUT_FILE, chunksize=50_000):
            for _, row in chunk.iterrows():
                answer = str(row["AnswerText"]).strip()

                if answer and answer.lower() != "nan":
                    out_file.write(f"{answer}\n\n")

    print("Documents created successfully")

if __name__ == "__main__":
    csv_to_documents()
