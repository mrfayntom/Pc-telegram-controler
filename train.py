import json
import os
import re
import copy
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

with open(r"path of your json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

commands = [item["command"] for item in raw_data]
intents = [item["intent"] for item in raw_data]

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(intents)

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

vocab_counter = Counter()
for sentence in commands:
    vocab_counter.update(tokenize(sentence))

word_to_index = {word: idx + 2 for idx, (word, _) in enumerate(vocab_counter.items())}
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1

def encode_sentence(sentence):
    return torch.tensor([word_to_index.get(word, 1) for word in tokenize(sentence)], dtype=torch.long)

class IntentDataset(Dataset):
    def __init__(self, sentences, labels):
        self.inputs = [encode_sentence(sentence) for sentence in sentences]
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def collate(batch):
    sequences, targets = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(targets)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    commands, encoded_labels, test_size=0.2, random_state=42
)

train_data = IntentDataset(train_sentences, train_labels)
test_data = IntentDataset(test_sentences, test_labels)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=collate)

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(combined))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntentClassifier(
    vocab_size=len(word_to_index),
    embed_dim=64,
    hidden_dim=128,
    output_dim=len(encoder.classes_)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss = float("inf")
wait = 0
patience = 3
backup = copy.deepcopy(model.state_dict())

for epoch in range(30):
    model.train()
    total = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item()

    model.eval()
    val_total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            val_total += loss_fn(preds, y_batch).item()

    avg_val = val_total / len(test_loader)
    print(f"Epoch {epoch+1}: Train Loss = {total:.4f}, Val Loss = {avg_val:.4f}")

    if avg_val < best_loss:
        best_loss = avg_val
        backup = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Stopped early.")
            break

model.load_state_dict(backup)
model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        predicted = torch.argmax(output, dim=1).cpu().numpy()
        predictions.extend(predicted)
        actuals.extend(y_batch.numpy())

print()
print(classification_report(actuals, predictions, target_names=encoder.classes_))

folder = r"path where you are saving the pt model"
os.makedirs(folder, exist_ok=True)

torch.save(model.state_dict(), os.path.join(folder, "intent_model.pt"))
with open(os.path.join(folder, "label_encoder.json"), "w") as f:
    json.dump(encoder.classes_.tolist(), f)
with open(os.path.join(folder, "word2idx.json"), "w") as f:
    json.dump(word_to_index, f)

print(f"\nModel and metadata saved to: {folder}")
