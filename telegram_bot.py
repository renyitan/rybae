from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.message import Message
import tensorflow as tf
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# telegram_token = "5087727847:AAHd1Sqc2WZf1r4AITGZJ8be4bl3mWWhNMQ" #rybae
telegram_token = "5037298729:AAGk1aaYg9-wjSeMak5Cvw0RxqgLWADwPUA"  # ryabe_dev

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
model = TFAutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')


def start_command(update, context):
    context.chat_data['chat_history_ids'] = []
    update.message.reply_text("rybae at your service")


def reset_command(update, context):
    context.chat_data['chat_history_ids'] = []
    update.message.reply_text('farewell')


def self_decorator(self, func):
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)
    return command_func


def message(self, update, context):
    user_message = update.message.text
    print('user: ', user_message)

    if 'chat_history_ids' not in context.chat_data:
        context.chat_data['chat_history_ids'] = None

    new_user_input_ids = tokenizer.encode(
        user_message + tokenizer.eos_token, return_tensors='tf')

    bot_input_ids = tf.concat(
        [context.chat_data['chat_history_ids'], new_user_input_ids], axis=-1) if context.chat_data['chat_history_ids'] != None else new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=2000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=1
    )

    context.chat_data['chat_history_ids'] = chat_history_ids

    bot_message = "{}".format(tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    print('bot_message: ', bot_message)
    update.message.reply_text(bot_message)


class TelegramBot:
    def __init__(self) -> None:
        self.updater = Updater(token=telegram_token, use_context=True)

        dispatcher = self.updater.dispatcher
        dispatcher.add_handler(CommandHandler('start', start_command))
        dispatcher.add_handler(CommandHandler('reset', reset_command))
        dispatcher.add_handler(MessageHandler(
            Filters.text, self_decorator(self, message)))

    def run(self):
        print("Running the telegram bot...")
        self.updater.start_polling()


tb = TelegramBot()
tb.run()
