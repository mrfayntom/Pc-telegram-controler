# PC Controller via Telegram

This project allows you to control your Windows PC remotely using your phone via a Telegram bot. It uses a trained intent classification model to interpret natural language commands and execute the corresponding system-level functions on your computer.

## How It Works

1. A message is sent to the Telegram bot.
2. The message is passed through a trained LSTM-based model to predict the user's intent.
3. Based on the predicted intent, a specific Python function is triggered to perform the desired task (e.g., locking the PC, taking a screenshot).
4. The bot sends a response back to the user via Telegram.

## Features

- Lock the PC
- Get battery status
- Get IP address
- Take a screenshot and receive it
- List folders and files
- Send selected files
- Open File Explorer or Chrome remotely

## Dataset (intent_class.json)

The `intent_class.json` file contains the training data used to teach the model how to classify user commands into intents.

Each entry in the JSON file is structured as:

`{
  "command": "lock my computer",
  "intent": "lock the pc"
}`

## Setup
- Requirements
```pip install torch python-telegram-bot psutil pyautogui```

## Telegram Bot Setup

1. Create a bot using [@BotFather](https://t.me/BotFather) on Telegram.
2. Copy the API token provided by BotFather.
3. In your script, replace the placeholder value with your actual Telegram bot token.

## File Structure

Ensure the following files are generated after training and are placed in accessible locations:

- `intent_model.pt`: Trained PyTorch model
- `word2idx.json`: Vocabulary dictionary used for tokenization
- `label_encoder.json`: Label encoder mapping for classifying intents

Update the script with the correct file paths so that these resources can be loaded at runtime.

## Security Considerations

- Do not expose your Telegram bot token or hardcoded system file paths in public repositories or shared environments.
- Use environment variables or a `.env` file to securely manage sensitive configuration values.
- If deploying this project in a shared or production environment, ensure proper authentication, access control, and network security are in place.

# Notice

I want to share an important update regarding my current availability and mental focus.

My board exam results are expected to be announced between **10 to 15 May 2025**. This period has been extremely stressful for me due to a recently introduced CBSE policy that could negatively impact students' final scores. Because of this change, there is a real possibility that the results may not reflect our true efforts, which has become a major concern for many students, including myself.

I have been aiming for **above 80%**, as that is the minimum requirement for a scholarship opportunity I’ve been working towards for a long time. However, the uncertainty surrounding the results has affected my mental state. As a result, I am currently experiencing a creative block and struggling to focus on programming or come up with new ideas.

If I am able to secure above 80% in the results, I expect to regain stability and become consistent with my work again.

Thank you for understanding.

