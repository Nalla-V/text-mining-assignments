from collections import Counter

# ========== Load IOB files ==========
def load_iob(path):
    sents = []
    tokens, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sents.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2: continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sents.append({"tokens": tokens, "tags": tags})
    return sents

# ========== Count entities ==========
def count_entities(sents):
    cnt = Counter()
    current = None
    for s in sents:
        for tag in s["tags"]:
            if tag.startswith("B-"):
                if current:
                    cnt[current] += 1
                current = tag[2:]
            elif tag.startswith("I-") and current == tag[2:]:
                continue
            else:
                if current:
                    cnt[current] += 1
                current = None
        if current:
            cnt[current] += 1
            current = None
    return cnt

splits = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

# ========== Print statistics ==========
for name, file in splits.items():
    data = load_iob(file)
    ent = count_entities(data)
    total_tokens = sum(len(s["tokens"]) for s in data)
    o_count = sum(1 for s in data for t in s["tags"] if t == "O")
    o_percent = o_count / total_tokens * 100 if total_tokens else 0
    total_ent = sum(ent.values())

    print(f"\n=== {name.upper()} ===")
    print(f"Sentences : {len(data):,}")
    print(f"Tokens    : {total_tokens:,}")
    print(f"Entities  : {total_ent:,}")
    print(f" % O tokens: {o_percent:5.1f}%")
    for typ in "ART CON LOC MAT PER SPE".split():
        c = ent.get(typ, 0)
        p = c / total_ent * 100 if total_ent else 0
        print(f"  {typ} : {c:4} ({p:5.1f}%)")