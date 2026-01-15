import random

INPUT_FILE = "data/processed/documents.txt"
OUTPUT_FILE = "data/processed/documents_sampled_clean.txt"
SAMPLE_RATIO = 0.05


def clean_text(text):
    text = text.strip()
    text = " ".join(text.split())
    return text


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

sample_size = int(SAMPLE_RATIO * len(lines))
sampled_lines = random.sample(lines, sample_size)

cleaned_lines = []

for line in sampled_lines:
    cleaned = clean_text(line)
    if cleaned:
        cleaned_lines.append(cleaned + "\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print(f"Clean 5% sampled dataset saved to {OUTPUT_FILE}")
