"""MS DialoGPT without fine-tuning and no response ranking"""

import os
from utils.model_utils import *
import functools
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')


def start_bot(update, context):
    context.chat_data['turns'] = []
    update.message.reply_text("rybae at your service")


def reset_bot(update, context):
    context.chat_data['chat_history_ids'] = []
    update.message.reply_text('it\'s your loss')


def get_chat_id(update, context):
    chat_id = -1,
    if(update.message is not None):
        # text message
        chat_id = update.message.chat.id
    elif update.callback_query is not None:
        # callback message
        chat_id = update.callback_query.message.chat.id
    elif update.poll is not None:
        # answer in poll
        chat_id = context.bot_data[update.poll.id]
    return chat_id


def reply_message(self, update, context):
    """Receive message, generate response, and reply."""

    max_turns_history = 2
    max_chat_context_history = 10

    # when first started, turns will be empty, create empty list
    if 'turns' not in context.chat_data:
        context.chat_data['turns'] = []
    turns = context.chat_data['turns']

    if max_turns_history == 0:
        context.chat_data['turns'] = []

    # extract incoming user message
    user_message = update.message.text

    # set chat action to "Typing..."
    context.bot.send_chat_action(chat_id=get_chat_id(
        update, context), action=ChatAction.TYPING)

    # parameters for generator_pipeline
    generator_pipeline_kwargs = {
        "max_length": 2000,
        "pad_token_id": self.generator_pipeline.tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 1
    }

    # establish current turn
    turn = {
        'user_messages': [],
        'bot_messages': []
    }

    turns.append(turn)
    turn['user_messages'].append(user_message)

    # merge turns into a single prompt
    current_prompt = ""
    from_index = max(len(turns) - max_turns_history - 1,
                     0) if max_turns_history >= 0 else 0
    for turn in turns[from_index:]:
        # each turn begins with user message to capture context
        for user_message in turn['user_messages']:
            current_prompt += strip_text(user_message) + \
                self.generator_pipeline.tokenizer.eos_token
        # add bot messages to capture context
        for bot_message in turn['bot_messages']:
            current_prompt += strip_text(bot_message) + \
                self.generator_pipeline.tokenizer.eos_token

    # generate the response
    bot_messages = generate_responses(
        user_message + self.generator_pipeline.tokenizer.eos_token,
        self.generator_pipeline,
        **generator_pipeline_kwargs
    )

    # reply with the first response
    if(len(bot_messages) == 1):
        bot_message = bot_messages[0]

    # append the bot message (response) into chat context
    turn['bot_messages'].append(bot_message)

    # only keep the most recent 5 chats to prevent storing too much
    if len(context.chat_data['turns']) >= max_chat_context_history:
        context.chat_data["turns"] = turns[1:]

    # reply the user
    update.message.reply_text(bot_message)


class RybaeBot:
    def __init__(self) -> None:
        print("Initialising rybae...")
        self.updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
        # self.PORT = int(os.environ.get('PORT'))
        # self.BASE_URL = os.environ.get('BASE_URL)

        self.generator_pipeline = load_pipeline(
            'text-generation', device=-1, model="microsoft/DialoGPT-large")

        dispatcher = self.updater.dispatcher
        dispatcher.add_handler(CommandHandler('start', start_bot))
        dispatcher.add_handler(CommandHandler('reset', reset_bot))

        # functools.partial allow us to add 'self' to callback 'reply_message'
        dispatcher.add_handler(MessageHandler(
            Filters.text, functools.partial(reply_message, self)))

    def run(self):
        print("Rybae is alive! 🚀")
        self.updater.start_polling()
        # self.updater.start_webhook(listen="0.0.0.0", port=self.PORT,
        #                            url_path=TELEGRAM_TOKEN, webhook_url=self.BASE_URL + TELEGRAM_TOKEN)

        self.updater.idle()
