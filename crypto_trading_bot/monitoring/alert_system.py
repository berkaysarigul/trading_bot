import requests

def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram alert error: {e}")

def send_discord_alert(webhook_url, message):
    data = {"content": message}
    try:
        requests.post(webhook_url, json=data)
    except Exception as e:
        print(f"Discord alert error: {e}") 