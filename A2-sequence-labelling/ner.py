import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from sklearn.metrics import classification_report

model_checkpoint = "bert-base-cased"
train_path = "train.txt"
val_path = "val.txt"
test_path = "test.txt"
output_dir = "bert-finetuned-ner"
num_train_epochs = 3
per_device_train_batch_size = 8

# ========== Parse IOB files ==========
def read_iob(path):
    sents  = []
    tokens = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sents.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
                continue

            parts = line.split()
            if len(parts) >= 3:
                token, tag = parts[0], parts[-1]
            elif len(parts) == 2:
                token, tag = parts
            else:
                continue
            tokens.append(token)
            labels.append(tag)

    if tokens:
        sents.append({"tokens": tokens, "ner_tags": labels})
    return sents

train_sents = read_iob(train_path)
val_sents = read_iob(val_path)
test_sents = read_iob(test_path)

# ========== Build label mappings ==========
all_labels = sorted({lab for s in (train_sents + val_sents + test_sents) for lab in s["ner_tags"]})
label2id = {lab: i for i, lab in enumerate(all_labels)}
id2label = {i: lab for lab, i in label2id.items()}
label_names = [id2label[i] for i in range(len(id2label))]

def convert_to_ids(example):
    example["ner_tag_ids"] = [label2id[l] for l in example["ner_tags"]]
    return example

train_sents = [convert_to_ids(s) for s in train_sents]
val_sents = [convert_to_ids(s) for s in val_sents]
test_sents = [convert_to_ids(s) for s in test_sents]

hf_datasets = DatasetDict({
    "train": Dataset.from_list(train_sents),
    "validation": Dataset.from_list(val_sents),
    "test": Dataset.from_list(test_sents),
})

# ========== Tokenizer & alignment ==========
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert tokenizer.is_fast, "This requires a fast tokenizer."

def align_labels_with_tokens(label_ids, word_ids):
    aligned = []
    prev = None
    for w in word_ids:
        if w is None:
            aligned.append(-100)
        elif w != prev:
            aligned.append(label_ids[w])
        else:
            aligned.append(-100)
        prev = w
    return aligned

def tokenize_and_align_labels(batch):
    tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    new_labels = []
    for i, labels in enumerate(batch["ner_tag_ids"]):
        word_ids = tokenized.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized["labels"] = new_labels
    return tokenized

tokenized_datasets = hf_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["tokens", "ner_tags", "ner_tag_ids"]
)

# ========== Data collator and model ==========
data_collator = DataCollatorForTokenClassification(tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id
)

# ========== Metrics ============
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    # Convert to string labels (ignore -100)
    true_labels = [[label_names[l] for l in sent if l != -100] for sent in labels]
    pred_labels = [
        [label_names[p] for p, l in zip(psent, tsent) if l != -100]
        for psent, tsent in zip(preds, labels)
    ]

    # seqeval: full entity spans
    seq_results = metric.compute(predictions=pred_labels, references=true_labels, zero_division=0)

    # Token-level (B-/I-)
    flat_true = [label_names[l] for sent in labels for l in sent if l != -100]
    flat_pred = [label_names[p] for psent, tsent in zip(preds, labels)
                 for p, l in zip(psent, tsent) if l != -100]
    token_report = classification_report(flat_true, flat_pred, labels=label_names,
                                         output_dict=True, zero_division=0)

    # Table 1: Token-level
    token_table = {}
    for lbl in label_names:
        token_table[lbl] = {
            "P": token_report[lbl]["precision"],
            "R": token_report[lbl]["recall"],
            "F1": token_report[lbl]["f1-score"]
        }

    # Table 2: Entity-level
    entity_table = {}
    for typ in seq_results:
        if typ not in {"overall_precision", "overall_recall", "overall_f1", "overall_accuracy"}:
            entity_table[typ] = {
                "P": seq_results[typ]["precision"],
                "R": seq_results[typ]["recall"],
                "F1": seq_results[typ]["f1"]
            }

    # Macro & micro
    micro_f1 = seq_results.get("overall_f1", 0.0)
    macro_f1 = np.mean([seq_results[t]["f1"] for t in entity_table]) if entity_table else 0.0

    return {
        "precision": seq_results.get("overall_precision", 0.0),
        "recall": seq_results.get("overall_recall", 0.0),
        "f1": seq_results.get("overall_f1", 0.0),
        "token_level": token_table,
        "entity_level": entity_table,
        "macro_entity_f1": macro_f1,
        "micro_entity_f1": micro_f1,
    }

# ========== TrainingArguments =========
training_args = TrainingArguments(
    output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# ========== Trainer ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ========== Train ====================
trainer.train()

# ========== Test Evaluation ==========
print("Running final evaluation on test set...")
test_result = trainer.predict(tokenized_datasets["test"])
test_metrics = compute_metrics((test_result.predictions, test_result.label_ids))

def print_assignment_tables(metrics):
    print("\n1. Precision / Recall / F1 per token label (B- / I-)")
    print("-"*80)
    header = f"{'Label':<12} {'P':>8} {'R':>8} {'F1':>8}"
    print(header)
    print("-"*len(header))
    for lbl, vals in metrics["token_level"].items():
        print(f"{lbl:<12} {vals['P']:8.4f} {vals['R']:8.4f} {vals['F1']:8.4f}")

    print("\n2. Precision / Recall / F1 per entity type (full spans)")
    print("-"*80)
    print(header)
    print("-"*len(header))
    for typ, vals in metrics["entity_level"].items():
        print(f"{typ:<12} {vals['P']:8.4f} {vals['R']:8.4f} {vals['F1']:8.4f}")

    print("\n3. Macro- and Micro-averaged F1 over entities")
    print("-"*80)
    print(f"{'Micro F1':<12} {metrics['micro_entity_f1']:8.4f}")
    print(f"{'Macro F1':<12} {metrics['macro_entity_f1']:8.4f}")

print_assignment_tables(test_metrics)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)