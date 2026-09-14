# get_corpus.py
import urllib.request
import re

DATASETS = {
    "1": ("The War of the Worlds (H.G. Wells)", "https://www.gutenberg.org/files/36/36-0.txt", 340000),
    "2": ("Frankenstein (Mary Shelley)", "https://www.gutenberg.org/files/84/84-0.txt", 430000),
    "3": ("The Time Machine (H.G. Wells)", "https://www.gutenberg.org/files/35/35-0.txt", 195000),
    "4": ("Tiny Shakespeare (Karpathy)", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", 300000)
}

print("Select a pre-configured dataset:")
for k, (name, _, chars) in DATASETS.items():
    print(f" [{k}] {name} (~{chars:,} chars)")

choice = input("\nEnter choice (1-4) [default 1]: ").strip()
if choice not in DATASETS:
    choice = "1"

name, url, target_len = DATASETS[choice]
print(f"\nDownloading '{name}'...")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    raw_text = response.read().decode('utf-8', errors='ignore')

# Strip Project Gutenberg boilerplate metadata if present
if "*** START OF" in raw_text:
    raw_text = raw_text.split("*** START OF")[1]
if "*** END OF" in raw_text:
    raw_text = raw_text.split("*** END OF")[0]

# Expand numbers to preserve semantic structure
digit_map = {
    '0': ' zero ', '1': ' one ', '2': ' two ', '3': ' three ', '4': ' four ',
    '5': ' five ', '6': ' six ', '7': ' seven ', '8': ' eight ', '9': ' nine '
}
for d, word in digit_map.items():
    raw_text = raw_text.replace(d, word)

# Filter strictly to the 32-token vocabulary: a-z, space, and .,!?-
allowed = set("abcdefghijklmnopqrstuvwxyz .,!?-")
sanitized = "".join([c for c in raw_text.lower().replace("\n", " ") if c in allowed])
sanitized = re.sub(' +', ' ', sanitized).strip()

# Slice to target range (up to 400k characters)
sanitized = sanitized[:400000]

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(sanitized)

print(f"Successfully processed and saved {len(sanitized):,} characters to 'corpus.txt'.")