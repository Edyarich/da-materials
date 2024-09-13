import requests
import sys

from flask import Flask, request
from waitress import serve
import logging
import telegram
from telegram import Update
from settings import *

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

app = Flask(__name__)
bot = telegram.Bot(BOT_TOKEN)

@app.route(f"/{BOT_TOKEN}", methods=['POST'])
def echo() -> dict:
    update = Update.de_json(request.get_json(force=True), bot)

    chat_id = update.message.chat.id
    text = update.message.text
    
    try:
        bot.sendMessage(chat_id=chat_id, text=text)
    except Exception:
        bot.sendMessage(chat_id=chat_id, text="Wrong input")

    return {'ok': True}

if __name__ == '__main__':
    ngrok_url = sys.argv[1]
    
    bot.setWebhook('{URL}/{HOOK}'.format(URL=ngrok_url, HOOK=BOT_TOKEN))
    
    serve(app, host=HOST, port=PORT)
