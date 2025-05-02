import os
import json
import torch
import ctypes
import socket
import psutil
import datetime
import pyautogui
import torch.nn as nn
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

model_file = r""  # Path to your model
vocab_file = r""  # Path to word2idx.json
labels_file = r""  # Path to label_encoder.json
images_dir = r""  # Directory containing folders with files

# === Load vocab and labels ===
with open(vocab_file, "r") as f:
    word2idx = json.load(f)

with open(labels_file, "r") as f:
    label_names = json.load(f)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        out = torch.cat((h[0], h[1]), dim=1)
        return self.fc(out)

model = SimpleLSTM(len(word2idx), 64, 128, len(label_names))
model.load_state_dict(torch.load(model_file, map_location="cpu"))
model.eval()

def tokenize(text):
    return text.lower().split()

def to_tensor(text):
    tokens = tokenize(text)
    indices = [word2idx.get(word, 0) for word in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def get_intent(text):
    x = to_tensor(text)
    with torch.no_grad():
        output = model(x)
        idx = torch.argmax(output, dim=1).item()
    return label_names[idx] if idx < len(label_names) else "unknown"

def lock_screen():
    ctypes.windll.user32.LockWorkStation()

def battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery info not available."
    status = f"Battery: {battery.percent}%"
    return status + ("\nCharging" if battery.power_plugged else "\nNot charging")

def pc_ip():
    try:
        return f"IP Address: {socket.gethostbyname(socket.gethostname())}"
    except:
        return "Unable to fetch IP."

def save_screenshot():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(r"", f"screenshot_{now}.png")  # Folder path blanked
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    return path

session = {"folders": [], "files": [], "current": None}

def show_folders():
    folders = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
    session["folders"] = folders
    return "\n".join(f"{i+1}. {name}" for i, name in enumerate(folders))

def show_files(folder_index):
    try:
        folder = session["folders"][folder_index - 1]
        path = os.path.join(images_dir, folder)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        session["files"] = files
        session["current"] = path
        return "\n".join(f"{i+1}. {name}" for i, name in enumerate(files))
    except IndexError:
        return "Invalid folder number."

async def send_file(index, update: Update):
    try:
        file_name = session["files"][index - 1]
        path = os.path.join(session["current"], file_name)
        await update.message.reply_document(document=open(path, "rb"))
    except:
        await update.message.reply_text("Couldn't send file.")

bot_token = ""  # Put your Telegram Bot Token here

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    intent = get_intent(text)

    if text.lower() == "show images":
        await update.message.reply_text(show_folders())
        return

    if text.isdigit():
        idx = int(text)
        if not session["current"]:
            response = show_files(idx)
            await update.message.reply_text(response)
        else:
            await send_file(idx, update)
        return

    if intent == "lock the pc":
        lock_screen()
        await update.message.reply_text("PC locked.")
    elif intent == "get pc battery":
        await update.message.reply_text(battery_status())
    elif intent == "get pc ip":
        await update.message.reply_text(pc_ip())
    elif intent == "take screenshot":
        path = save_screenshot()
        await update.message.reply_photo(photo=open(path, "rb"))
    elif intent == "open file explorer":
        os.startfile("explorer")
        await update.message.reply_text("Explorer opened.")
    elif intent == "open chrome":
        os.startfile(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        await update.message.reply_text("Chrome opened.")
    else:
        await update.message.reply_text(f"Intent: {intent}")

def main():
    app = Application.builder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
