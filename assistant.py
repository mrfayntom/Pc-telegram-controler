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


MODEL_PATH = r""  # model file path
WORD2IDX_PATH = r""  # vocab file path
LABEL_ENCODER_PATH = r""  # label names path
BASE_DIR = r""  # directory for image folders


with open(WORD2IDX_PATH, "r") as f:
    word2idx = json.load(f)

with open(LABEL_ENCODER_PATH, "r") as f:
    label_encoder = json.load(f)

vocab_size = len(word2idx)
embed_dim = 64
hidden_dim = 128
num_classes = len(label_encoder)


class IntentLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h)

model = IntentLSTMModel(vocab_size, embed_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def tokenize(text):
    return text.lower().split()

def vectorize(sentence):
    tokens = tokenize(sentence)
    idxs = [word2idx.get(word, 0) for word in tokens]
    return torch.tensor(idxs, dtype=torch.long).unsqueeze(0)

def predict_intent(sentence):
    x = vectorize(sentence)
    with torch.no_grad():
        output = model(x)
        predicted_idx = torch.argmax(output, dim=1).item()
    if predicted_idx < len(label_encoder):
        return label_encoder[predicted_idx]
    return "unknown"


def lock_pc():
    ctypes.windll.user32.LockWorkStation()

def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery info not available."
    info = f"Battery Percentage: {battery.percent}%"
    info += "\nLaptop is charging." if battery.power_plugged else "\nLaptop is not charging."
    return info

def get_pc_ip():
    try:
        ip = socket.gethostbyname(socket.gethostname())
        return f"Your PC's IP Address: {ip}"
    except:
        return "Unable to get IP address."

def take_screenshot():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(r"", f"screenshot_{now}.png")  # target path here
    image = pyautogui.screenshot()
    image.save(path)
    return path


session = {
    "folder_list": [],
    "current_folder": None,
    "file_list": []
}

def list_folders(base_path):
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    session["folder_list"] = folders
    response = "Folders:\n"
    for i, folder in enumerate(folders):
        response += f"{i + 1}. {folder}\n"
    return response

def list_files(folder_index):
    try:
        folder = session["folder_list"][folder_index - 1]
        full_path = os.path.join(BASE_DIR, folder)
        files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
        session["current_folder"] = full_path
        session["file_list"] = files
        response = f"Files in '{folder}':\n"
        for i, file in enumerate(files):
            response += f"{i + 1}. {file}\n"
        return response
    except IndexError:
        return "Invalid folder number."

async def send_file(file_index, update: Update):
    try:
        filename = session["file_list"][file_index - 1]
        file_path = os.path.join(session["current_folder"], filename)
        await update.message.reply_document(document=open(file_path, "rb"))
    except IndexError:
        await update.message.reply_text("Invalid file number.")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")


BOT_TOKEN = ""  # Your bot token here

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.message.text
    print(f"[{user.first_name}] -> {msg}")

    intent = predict_intent(msg)
    print(f"Predicted intent: {intent}")

    if msg.lower() == "show images":
        response = list_folders(BASE_DIR)
        await update.message.reply_text(response)
        return

    if msg.strip().isdigit():
        num = int(msg.strip())
        if not session["current_folder"]:
            response = list_files(num)
            await update.message.reply_text(response)
        else:
            await send_file(num, update)
        return

    if intent.lower() == "lock the pc":
        lock_pc()
        response = "Your PC is being locked."
    elif intent.lower() == "get pc battery":
        response = get_battery_status()
    elif intent.lower() == "get pc ip":
        response = get_pc_ip()
    elif intent.lower() == "take screenshot":
        screenshot_path = take_screenshot()
        await update.message.reply_text(f"Screenshot saved at: {screenshot_path}")
        await update.message.reply_photo(photo=open(screenshot_path, 'rb'))
        return
    elif intent.lower() == "open file explorer":
        os.startfile("explorer")
        response = "File Explorer opened."
    elif intent.lower() == "open chrome":
        os.startfile(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        response = "Chrome browser opened."
    else:
        response = f"Intent: {intent}"

    await update.message.reply_text(response)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
