# 📁 telegram_bot.py — слой представления (интерфейс Telegram)
import logging
from telegram import Update, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from rag_agent import get_rag_chain
from db_utils import add_documents
from user_db import init_user_db, check_credentials
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Константы состояний ---
LOGIN, PASSWORD, AUTHED, ROLE_USER = range(4)
ROLE_ADMIN = 0
ROLE_USER_INT = 1

# --- Telegram Handlers ---
auth_users = {}
rag_chain = get_rag_chain()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_username = update.effective_user.username
    logger.info(f"/start от {update.effective_user.id}, username: {tg_username}")
    if tg_username:
        context.user_data["username"] = tg_username
        await update.message.reply_text("Теперь введи пароль:")
        return PASSWORD
    else:
        await update.message.reply_text("У тебя не задано имя пользователя в Telegram. Введи имя вручную:")
        return LOGIN

async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["username"] = update.message.text
    logger.info(f"Пользователь ввёл имя: {update.message.text}")
    await update.message.reply_text("Теперь введи пароль:")
    return PASSWORD

async def password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = context.user_data.get("username")
    password = update.message.text
    valid, role = check_credentials(username, password)
    if valid:
        auth_users[update.effective_user.id] = role
        role_label = "admin" if role == ROLE_ADMIN else "user"
        logger.info(f"Пользователь {username} вошёл как {role_label}")
        await update.message.reply_text(f"Успешный вход как {role_label}. Теперь можешь задавать вопросы или отправлять документы (если ты админ).")
        return ROLE_USER
    else:
        logger.warning(f"Неудачная попытка входа: {username}")
        await update.message.reply_text("Неверный логин или пароль. Попробуй ещё раз /start")
        return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in auth_users:
        logger.info(f"Неавторизованный запрос от {user_id}")
        await update.message.reply_text("Сначала авторизуйтесь командой /start")
        return

    role = auth_users[user_id]
    if update.message.document and role == ROLE_ADMIN:
        file = update.message.document
        if file.mime_type == "application/pdf":
            logger.info(f"Загрузка PDF от {user_id}: {file.file_name}")
            file_path = await file.get_file()
            local_path = f"temp/{file.file_name}"
            os.makedirs("temp", exist_ok=True)
            await file_path.download_to_drive(local_path)
            loader = PyPDFLoader(local_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(loader.load())
            logger.info(f"Разбито на {len(docs)} фрагментов. Добавление в базу...")
            add_documents(docs)
            logger.info("Документ успешно добавлен в базу знаний")
            await update.message.reply_text("📄 Документ успешно добавлен в базу знаний.")
            os.remove(local_path)
        else:
            logger.warning(f"Файл не PDF от {user_id}")
            await update.message.reply_text("⚠️ Принимаются только PDF-файлы.")
    elif update.message.text:
        query = update.message.text
        logger.info(f"Вопрос от {user_id}: {query}")
        response = rag_chain.invoke({"question": query})
        await update.message.reply_text(response.strip())
    else:
        logger.warning(f"Нераспознанный ввод от {user_id}")
        await update.message.reply_text("Отправь текстовый вопрос или PDF-файл.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Сессия отменена пользователем {update.effective_user.id}")
    await update.message.reply_text("Окей, выход из сессии.")
    return ConversationHandler.END

# --- Запуск бота ---
def main():
    logger.info("Запуск Telegram-бота...")
    init_user_db()
    try:
        with open("bot_token.txt", "r", encoding="utf-8") as f:
            token = f.read().strip()
    except FileNotFoundError:
        logger.error("Файл bot_token.txt не найден")
        raise ValueError("Файл bot_token.txt не найден. Помести туда токен Telegram-бота")

    app = Application.builder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            LOGIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, login)],
            PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, password)],
            ROLE_USER: [MessageHandler(filters.ALL, handle_message)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(conv_handler)
    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
