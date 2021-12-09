from functools import wraps
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


from model_utils import *

# telegram_token = "5087727847:AAHd1Sqc2WZf1r4AITGZJ8be4bl3mWWhNMQ" #rybae
telegram_token = "5037298729:AAGk1aaYg9-wjSeMak5Cvw0RxqgLWADwPUA"  # ryabe_dev


def start_command(update, context):
    context.chat_data['turns'] = []
    update.message.reply_text("rybae at your service")


def reset_command(update, context):
    context.chat_data['chat_history_ids'] = []
    update.message.reply_text('farewell')


def self_decorator(self, func):
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)
    return command_func


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


def send_action(action):
    def send_action_decorator(fn):
        @wraps(fn)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=get_chat_id(
                update, context), action=action)
            return fn(self, update, context, *args, **kwargs)
        return command_func
    return send_action_decorator


@send_action(ChatAction.TYPING)  # decorator for message()
def message(self, update, context):
    """Receive message, generate response, and send it back to the user."""

    max_turns_history = 2
    max_chat_context_history = 10

    if 'turns' not in context.chat_data:
        context.chat_data['turns'] = []
    turns = context.chat_data['turns']

    # extract incoming user message
    user_message = update.message.text

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

    if max_turns_history == 0:
        context.chat_data['turns'] = []

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
            current_prompt += clean_text(user_message) + \
                self.generator_pipeline.tokenizer.eos_token
        # add bot messages to capture context
        for bot_message in turn['bot_messages']:
            current_prompt += clean_text(bot_message) + \
                self.generator_pipeline.tokenizer.eos_token

    # generate the response
    bot_messages = generate_responses(
        user_message + self.generator_pipeline.tokenizer.eos_token,
        self.generator_pipeline,
        **generator_pipeline_kwargs
    )

    if(len(bot_messages) == 1):
        bot_message = bot_messages[0]

    # append the bot message into chat context
    turn['bot_messages'].append(bot_message)

    # only keep the most recent 5 chats to prevent storing too much
    if len(context.chat_data['turns']) >= max_chat_context_history:
        context.chat_data["turns"] = turns[1:]

    # reply the user
    update.message.reply_text(bot_message)


class TelegramBot:
    def __init__(self) -> None:
        self.updater = Updater(token=telegram_token, use_context=True)

        self.generator_pipeline = load_pipeline(
            'text-generation', device=-1, model="microsoft/DialoGPT-medium")

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
