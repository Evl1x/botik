import asyncio
import base64
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional

import requests
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
from openai import OpenAI

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:
    ZoneInfo = None  # type: ignore

# =========================
# ENV / SETTINGS
# =========================
load_dotenv()

BOT_TOKEN = (os.getenv("BOT_TOKEN", "") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY", "") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()

EDAMAM_APP_ID = (os.getenv("EDAMAM_APP_ID", "") or "").strip()
EDAMAM_APP_KEY = (os.getenv("EDAMAM_APP_KEY", "") or "").strip()

# Каналы можно задать так: CHANNELS=@a,@b,@c
CHANNELS_RAW = (os.getenv("CHANNELS", "") or "").strip()
CHANNELS = [c.strip() for c in CHANNELS_RAW.split(",") if c.strip()] if CHANNELS_RAW else ["@Evl1xxx"]

# Куда предлагать продукт
SUGGEST_CHAT = "@Andreqq3"

# "Время пользователя" — берём Europe/Amsterdam (как в этом чате).
BOT_TZ_NAME = os.getenv("BOT_TZ", "Europe/Amsterdam").strip()  # можно переопределить в .env
BOT_TZ = ZoneInfo(BOT_TZ_NAME) if ZoneInfo else None

# =========================
# BOT INIT
# =========================
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# =========================
# FSM STATES
# =========================
class FoodStates(StatesGroup):
    waiting_for_food_name = State()
    waiting_for_food_weight = State()

class PhotoStates(StatesGroup):
    waiting_for_photo = State()
    waiting_confirm = State()

# =========================
# IN-MEMORY USER DATA
# =========================
# user_data[user_id] = { foods: [...], total_calories: float, date: "YYYY-MM-DD" }
user_data: Dict[int, Dict] = {}

def now_local() -> datetime:
    if BOT_TZ:
        return datetime.now(tz=BOT_TZ)
    return datetime.now()

def today_str() -> str:
    return now_local().strftime("%Y-%m-%d")

def get_user_data(user_id: int) -> Dict:
    """Создаём/обновляем дневник, если новый день."""
    if user_id not in user_data:
        user_data[user_id] = {"foods": [], "total_calories": 0.0, "date": today_str()}
        return user_data[user_id]

    if user_data[user_id].get("date") != today_str():
        user_data[user_id] = {"foods": [], "total_calories": 0.0, "date": today_str()}
    return user_data[user_id]

# =========================
# KEYBOARDS
# =========================
def kb_channels() -> InlineKeyboardMarkup:
    rows = []
    for ch in CHANNELS:
        uname = ch.replace("@", "")
        rows.append([InlineKeyboardButton(text=f"📢 Подписаться на {ch}", url=f"https://t.me/{uname}")])
    rows.append([InlineKeyboardButton(text="✅ Я подписался", callback_data="check_subscription")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍎 Добавить еду", callback_data="add_food")],
        [InlineKeyboardButton(text="📷 Еда по фото", callback_data="add_food_photo")],
        [InlineKeyboardButton(text="📊 Итоги за день", callback_data="show_stats")],
        [InlineKeyboardButton(text="🗑️ Очистить день", callback_data="clear_day")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="help")],
        [InlineKeyboardButton(text="💬 Поддержка", url=f"https://t.me/{SUGGEST_CHAT.replace('@','')}")],
    ])

def kb_back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="↩️ Назад в меню", callback_data="main_menu")]
    ])

def kb_weights() -> InlineKeyboardMarkup:
    weights = [50, 100, 150, 200, 250]
    rows = []
    for i in range(0, len(weights), 2):
        row = []
        for w in weights[i:i+2]:
            row.append(InlineKeyboardButton(text=f"{w}г", callback_data=f"weight:{w}"))
        rows.append(row)
    rows.append([InlineKeyboardButton(text="📝 Другой вес", callback_data="custom_weight")])
    rows.append([InlineKeyboardButton(text="↩️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_photo_confirm() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Добавить в дневник", callback_data="photo_confirm_add")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="photo_confirm_cancel")],
        [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")],
    ])

def suggest_url(product_text: str) -> str:
    text = f"Хочу добавить продукт/блюдо: {product_text}"
    encoded = urllib.parse.quote(text[:300])
    return f"https://t.me/{SUGGEST_CHAT.replace('@','')}?text={encoded}"

def kb_not_found(query: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💡 Предложить блюдо", url=suggest_url(query))],
        [InlineKeyboardButton(text="📷 Попробовать по фото", callback_data="add_food_photo")],
        [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")],
    ])

def kb_stats_menu(has_foods: bool) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="🍎 Добавить еду", callback_data="add_food")],
        [InlineKeyboardButton(text="📷 Еда по фото", callback_data="add_food_photo")],
    ]
    if has_foods:
        rows.insert(0, [InlineKeyboardButton(text="🗑️ Удалить один продукт", callback_data="delete_menu")])
    rows += [
        [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_delete_list(user_id: int) -> InlineKeyboardMarkup:
    foods = user_data.get(user_id, {}).get("foods", []) or []
    rows = []
    for idx, f in enumerate(foods):
        name = (f.get("name") or "Без названия")[:22]
        kcal = float(f.get("calories") or 0)
        rows.append([InlineKeyboardButton(text=f"❌ {idx+1}. {name} ({kcal:.0f}ккал)", callback_data=f"delete_one:{idx}")])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="show_stats")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

# =========================
# SUBSCRIPTION CHECK
# =========================
async def is_subscribed(user_id: int) -> bool:
    if not CHANNELS:
        return True
    for ch in CHANNELS:
        try:
            member = await bot.get_chat_member(chat_id=ch, user_id=user_id)
            if member.status not in ("member", "administrator", "creator"):
                return False
        except:
            return False
    return True

# =========================
# QUERY NORMALIZATION (RU)
# =========================
STOP_WORDS = {
    "с", "без", "и", "на", "в", "из", "для", "или", "по",
    "сахаром", "молоком", "лимоном", "медом", "солью",
    "вареный", "вареная", "варёный", "варёная",
    "жареный", "жареная", "запеченный", "запеченная",
    "черный", "чёрный", "зеленый", "зелёный",
    "светлое", "темное", "тёмное",
    "бутылка", "банка", "стакан", "чашка", "порция",
}

RU_SYNONYMS = {
    "чай": "tea",
    "черный чай": "black tea",
    "чёрный чай": "black tea",
    "зеленый чай": "green tea",
    "зелёный чай": "green tea",
    "кофе": "coffee",
    "пиво": "beer",
    "вода": "water",
    "сок": "juice",
    "гречка": "buckwheat",
    "гречневая каша": "buckwheat",
    "рис": "rice",
    "макароны": "pasta",
    "курица": "chicken",
    "куриная грудка": "chicken breast",
}

def normalize_ru_query(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[,;:()]+", " ", q)
    q = re.sub(r"\s+", " ", q)

    # убираем "500мл", "200 г", "0.5 л"
    q = re.sub(r"\b\d+([.,]\d+)?\s*(г|гр|кг|мл|л)\b", " ", q)
    q = re.sub(r"\b\d+([.,]\d+)?\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    parts = [w for w in q.split() if w not in STOP_WORDS]
    q2 = " ".join(parts).strip()
    return q2 or q

# =========================
# EDAMAM SEARCH (async via to_thread)
# =========================
def _edamam_request(query: str) -> Optional[Dict]:
    url = "https://api.edamam.com/api/food-database/v2/parser"
    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": query,
        "nutrition-type": "cooking"
    }
    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json()
    hints = data.get("hints") or []
    if not hints:
        return None

    food = hints[0].get("food") or {}
    nutrients = food.get("nutrients") or {}
    kcal = float(nutrients.get("ENERC_KCAL", 0) or 0)

    return {
        "name": food.get("label", query),
        "calories": kcal,
        "protein": float(nutrients.get("PROCNT", 0) or 0),
        "fat": float(nutrients.get("FAT", 0) or 0),
        "carbs": float(nutrients.get("CHOCDF", 0) or 0),
    }

async def edamam_search(query: str) -> Optional[Dict]:
    try:
        return await asyncio.to_thread(_edamam_request, query)
    except Exception as e:
        logging.error(f"Edamam error: {e}")
        return None

def format_food_info(food_data: Dict) -> str:
    text = f"🍽 <b>{food_data['name']}</b>\n"
    text += f"🔥 <b>Калории:</b> {food_data['calories']:.1f} ккал/100г\n\n"
    text += "<b>📊 Состав на 100г:</b>\n"
    text += f"🥩 Белки: {food_data.get('protein', 0):.1f}г\n"
    text += f"🥑 Жиры: {food_data.get('fat', 0):.1f}г\n"
    text += f"🍚 Углеводы: {food_data.get('carbs', 0):.1f}г\n\n"
    return text

# =========================
# OPENAI HELPERS (JSON safe)
# =========================
def extract_json(text: str) -> Dict:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(t[s:e+1])
    raise ValueError("Не удалось извлечь JSON из ответа ИИ.")

def openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан в .env")
    return OpenAI(api_key=OPENAI_API_KEY)

def ai_suggest_terms_sync(query_ru: str) -> List[str]:
    client = openai_client()
    prompt = (
        "Пользователь ввёл продукт/блюдо по-русски. "
        "Нужно помочь найти это в международной базе еды (Edamam). "
        "Верни ТОЛЬКО JSON вида:\n"
        '{ "terms": ["term1","term2","term3","term4","term5"] }\n'
        "Правила: terms 3-7 коротких англ. вариантов, без лишнего текста.\n"
        f"Ввод: {query_ru}"
    )
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    data = extract_json(getattr(resp, "output_text", "") or "")
    terms = data.get("terms") or []
    return [str(t).strip() for t in terms if str(t).strip()]

def image_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def ai_food_from_photo_sync(image_bytes: bytes) -> Dict:
    client = openai_client()
    data_url = image_to_data_url(image_bytes)

    prompt = (
        "Ты распознаешь еду по фото. Верни ТОЛЬКО валидный JSON:\n"
        "{\n"
        '  "items":[{"name_ru":string,"name_en":string,"grams":number}],\n'
        '  "confidence": number,\n'
        '  "notes": string\n'
        "}\n"
        "Правила:\n"
        "- items 1-10 компонентов (рис, курица, салат, соус...).\n"
        "- grams: примерная масса каждого компонента.\n"
        "- name_en: короткое английское название для поиска в базе.\n"
        "- если не еда/неясно: items=[], confidence низкая.\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return extract_json(getattr(resp, "output_text", "") or "")

# =========================
# SMART SEARCH: RU -> Edamam -> AI terms -> Edamam
# =========================
async def smart_food_search(query_raw: str) -> Optional[Dict]:
    if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
        return None

    q = normalize_ru_query(query_raw)
    q = RU_SYNONYMS.get(q, q)

    # 1) прямая попытка
    res = await edamam_search(q)
    if res and res.get("calories", 0) > 0:
        return res

    # 2) если есть openai — пробуем варианты
    if not OPENAI_API_KEY:
        return None

    try:
        terms = await asyncio.to_thread(ai_suggest_terms_sync, query_raw)
    except Exception as e:
        logging.error(f"AI terms error: {e}")
        return None

    for term in terms[:8]:
        res2 = await edamam_search(term)
        if res2 and res2.get("calories", 0) > 0:
            return res2

    return None

# =========================
# DAILY SUMMARY 21:00
# =========================
def build_daily_report(user_id: int) -> str:
    ud = get_user_data(user_id)
    foods = ud.get("foods", []) or []
    total = float(ud.get("total_calories") or 0)
    date_human = now_local().strftime("%d.%m.%Y")

    if not foods:
        return (
            f"📊 <b>Итоги за {date_human}</b>\n\n"
            "Сегодня нет записей по еде.\n"
            "Завтра продолжим 💪"
        )

    lines = [f"📊 <b>Итоги за {date_human}</b>\n"]
    for i, f in enumerate(foods, 1):
        name = f.get("name", "Без названия")
        w = int(f.get("weight", 0) or 0)
        kcal = float(f.get("calories", 0) or 0)
        t = f.get("time", "")
        lines.append(f"{i}. {name} — {w}г ({kcal:.1f} ккал) {('в ' + t) if t else ''}".strip())

    lines.append(f"\n🔥 <b>Всего:</b> {total:.1f} ккал")
    lines.append("\nСпокойной ночи 🌙")
    return "\n".join(lines)

async def daily_summary_loop():
    """Каждый день в 21:00 (по BOT_TZ) отправляет итоги всем пользователям."""
    while True:
        try:
            now = now_local()
            target = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if target <= now:
                target = target + timedelta(days=1)

            sleep_seconds = (target - now).total_seconds()
            await asyncio.sleep(max(1, int(sleep_seconds)))

            # отправляем всем пользователям, кто есть в user_data
            for uid in list(user_data.keys()):
                try:
                    # если у юзера новый день — get_user_data сам обновит, но нам нужен отчет за текущий день:
                    # поэтому берем напрямую stored ud, без обновления даты
                    # (если дата уже сменилась, отчет будет пустым — ок)
                    text = build_daily_report(uid)
                    await bot.send_message(uid, text, parse_mode="HTML", reply_markup=kb_main())
                except Exception:
                    # пользователь мог заблокировать бота и т.д.
                    continue

        except Exception as e:
            logging.error(f"Daily summary loop error: {e}")
            await asyncio.sleep(10)

# =========================
# HANDLERS
# =========================
def main_menu_text() -> str:
    return (
        "🍏 <b>Калькулятор калорий</b>\n"
        "━━━━━━━━━━━━━━\n"
        "• Добавляй еду текстом\n"
        "• Или распознавай по фото 📷\n"
        "• Итоги дня автоматически в <b>21:00</b>\n\n"
        "Выбирай действие ниже 👇"
    )

@dp.message(Command("start"))
async def start_cmd(message: types.Message, state: FSMContext):
    await state.clear()
    uid = message.from_user.id

    if CHANNELS and not await is_subscribed(uid):
        await message.answer(
            "👋 <b>Добро пожаловать!</b>\n\n"
            "Чтобы пользоваться ботом — подпишитесь на канал:",
            reply_markup=kb_channels(),
            parse_mode="HTML"
        )
        return

    get_user_data(uid)
    await message.answer(main_menu_text(), reply_markup=kb_main(), parse_mode="HTML")

@dp.callback_query(F.data == "check_subscription")
async def check_sub_cb(callback: types.CallbackQuery):
    await callback.answer()
    uid = callback.from_user.id

    if not await is_subscribed(uid):
        await callback.message.edit_text(
            "❌ <b>Вы ещё не подписались!</b>\n\n"
            "Подпишитесь и нажмите «Я подписался».",
            reply_markup=kb_channels(),
            parse_mode="HTML"
        )
        return

    get_user_data(uid)
    await callback.message.edit_text(main_menu_text(), reply_markup=kb_main(), parse_mode="HTML")

@dp.callback_query(F.data == "main_menu")
async def main_menu(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.clear()
    await callback.message.edit_text(main_menu_text(), reply_markup=kb_main(), parse_mode="HTML")

@dp.callback_query(F.data == "help")
async def help_cb(callback: types.CallbackQuery):
    await callback.answer()
    text = (
        "❓ <b>Помощь</b>\n\n"
        "1) Нажми <b>Добавить еду</b> → введи название\n"
        "2) Если не нашлось — появится кнопка <b>Предложить блюдо</b>\n"
        "3) Нажми <b>Еда по фото</b> → отправь фото еды\n\n"
        "Авто-итоги приходят каждый день в <b>21:00</b>."
    )
    await callback.message.edit_text(text, reply_markup=kb_back_to_menu(), parse_mode="HTML")

# -------- TEXT ADD FLOW
@dp.callback_query(F.data == "add_food")
async def add_food_cb(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    uid = callback.from_user.id

    if CHANNELS and not await is_subscribed(uid):
        await callback.message.edit_text(
            "❌ <b>Нет подписки.</b>\nПодпишитесь на канал:",
            reply_markup=kb_channels(),
            parse_mode="HTML"
        )
        return

    await state.clear()
    await state.set_state(FoodStates.waiting_for_food_name)

    await callback.message.edit_text(
        "🔍 <b>Введите продукт или блюдо</b>\n\n"
        "Примеры: чай, пиво, гречка, куриная грудка, пельмени.\n"
        "<i>Если не найдётся — можно предложить блюдо кнопкой.</i>",
        reply_markup=kb_back_to_menu(),
        parse_mode="HTML"
    )

@dp.message(FoodStates.waiting_for_food_name, F.text)
async def food_name_msg(message: types.Message, state: FSMContext):
    query = (message.text or "").strip()
    if len(query) < 2:
        await message.answer("❌ Введите минимум 2 символа.")
        return

    wait = await message.answer("🔍 Ищу в базе...")
    food = await smart_food_search(query)
    await wait.delete()

    if not food:
        await state.clear()
        await message.answer(
            f"❌ Не нашёл: <b>{query}</b>\n\n"
            "Можешь предложить блюдо, и мы добавим его в базу запросов 👇",
            parse_mode="HTML",
            reply_markup=kb_not_found(query)
        )
        return

    await state.update_data(food_name=food["name"], calories_per_100=float(food["calories"]))
    await state.set_state(FoodStates.waiting_for_food_weight)

    text = "🌍 <b>Найдено:</b>\n" + format_food_info(food) + "Выберите вес порции:"
    await message.answer(text, reply_markup=kb_weights(), parse_mode="HTML")

@dp.callback_query(FoodStates.waiting_for_food_weight, F.data.startswith("weight:"))
async def weight_cb(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    w = int(callback.data.split(":")[1])

    data = await state.get_data()
    name = data.get("food_name")
    c100 = float(data.get("calories_per_100") or 0)

    if not name or c100 <= 0:
        await state.clear()
        await callback.message.edit_text("❌ Ошибка данных. Начните заново.", reply_markup=kb_main())
        return

    kcal = c100 * w / 100.0
    uid = callback.from_user.id
    ud = get_user_data(uid)
    ud["foods"].append({"name": name, "weight": w, "calories": kcal, "time": now_local().strftime("%H:%M")})
    ud["total_calories"] += kcal

    await state.clear()
    await callback.message.edit_text(
        f"✅ <b>Добавлено!</b>\n\n"
        f"🍽 <b>{name}</b>\n"
        f"⚖️ {w} г\n"
        f"🔥 {kcal:.1f} ккал\n\n"
        f"📊 Всего сегодня: {ud['total_calories']:.1f} ккал",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🍎 Добавить ещё", callback_data="add_food")],
            [InlineKeyboardButton(text="📷 Еда по фото", callback_data="add_food_photo")],
            [InlineKeyboardButton(text="📊 Итоги", callback_data="show_stats")],
            [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
        ])
    )

@dp.callback_query(FoodStates.waiting_for_food_weight, F.data == "custom_weight")
async def custom_weight_cb(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_text(
        "📝 <b>Введите вес в граммах</b>\n\nНапример: 175",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="↩️ Отмена", callback_data="main_menu")]
        ])
    )

@dp.message(FoodStates.waiting_for_food_weight, F.text)
async def custom_weight_msg(message: types.Message, state: FSMContext):
    t = (message.text or "").strip()
    if not t.isdigit():
        await message.answer("❌ Введите число, например 150.")
        return
    w = int(t)
    if w <= 0 or w > 5000:
        await message.answer("❌ Вес должен быть от 1 до 5000.")
        return

    data = await state.get_data()
    name = data.get("food_name")
    c100 = float(data.get("calories_per_100") or 0)
    if not name or c100 <= 0:
        await state.clear()
        await message.answer("❌ Ошибка данных. Начните заново.")
        return

    kcal = c100 * w / 100.0
    uid = message.from_user.id
    ud = get_user_data(uid)
    ud["foods"].append({"name": name, "weight": w, "calories": kcal, "time": now_local().strftime("%H:%M")})
    ud["total_calories"] += kcal

    await state.clear()
    await message.answer(
        f"✅ <b>Добавлено!</b>\n\n"
        f"🍽 <b>{name}</b>\n"
        f"⚖️ {w} г\n"
        f"🔥 {kcal:.1f} ккал\n\n"
        f"📊 Всего сегодня: {ud['total_calories']:.1f} ккал",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🍎 Добавить ещё", callback_data="add_food")],
            [InlineKeyboardButton(text="📷 Еда по фото", callback_data="add_food_photo")],
            [InlineKeyboardButton(text="📊 Итоги", callback_data="show_stats")],
            [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
        ])
    )

# -------- PHOTO FLOW
@dp.callback_query(F.data == "add_food_photo")
async def add_food_photo_cb(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    uid = callback.from_user.id

    if CHANNELS and not await is_subscribed(uid):
        await callback.message.edit_text(
            "❌ <b>Нет подписки.</b>\nПодпишитесь на канал:",
            reply_markup=kb_channels(),
            parse_mode="HTML"
        )
        return

    if not OPENAI_API_KEY:
        await callback.message.edit_text(
            "❌ <b>Фото-распознавание не настроено.</b>\n"
            "Добавьте OPENAI_API_KEY в .env",
            parse_mode="HTML",
            reply_markup=kb_back_to_menu()
        )
        return

    await state.clear()
    await state.set_state(PhotoStates.waiting_for_photo)

    await callback.message.edit_text(
        "📷 <b>Отправьте фото еды</b>\n\n"
        "Я разберу на продукты и посчитаю калории.\n"
        "<i>Лучше фото сверху, чтобы было видно порции.</i>",
        parse_mode="HTML",
        reply_markup=kb_back_to_menu()
    )

@dp.message(PhotoStates.waiting_for_photo, F.photo)
async def photo_msg(message: types.Message, state: FSMContext):
    uid = message.from_user.id
    if CHANNELS and not await is_subscribed(uid):
        await message.answer("❌ Нет подписки.", reply_markup=kb_channels(), parse_mode="HTML")
        await state.clear()
        return

    wait = await message.answer("🧠 Анализирую фото...")

    # 1) скачать фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    # 2) распознать через OpenAI
    try:
        ai_res = await asyncio.to_thread(ai_food_from_photo_sync, image_bytes)
    except Exception as e:
        logging.exception("PHOTO AI ERROR")
        await wait.delete()
        await state.clear()
        await message.answer(
            f"❌ Не смог распознать фото.\n\nОшибка: <code>{e}</code>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📷 Попробовать снова", callback_data="add_food_photo")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
            ])
        )
        return

    items = ai_res.get("items") or []
    conf = ai_res.get("confidence", 0)
    notes = ai_res.get("notes", "")

    if not items:
        await wait.delete()
        await state.clear()
        await message.answer(
            "🤔 <b>Не получилось выделить еду на фото.</b>\n\n"
            f"<i>{notes}</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📷 Попробовать снова", callback_data="add_food_photo")],
                [InlineKeyboardButton(text="🍎 Добавить текстом", callback_data="add_food")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
            ])
        )
        return

    # 3) каждый компонент -> Edamam (с умным поиском)
    enriched = []
    unknown = []
    total_kcal = 0.0

    for it in items[:12]:
        name_ru = str(it.get("name_ru", "") or "").strip()
        name_en = str(it.get("name_en", "") or "").strip()
        grams = it.get("grams", 0)
        try:
            grams = int(float(grams))
        except:
            grams = 100
        if grams <= 0:
            grams = 100

        q = name_en or name_ru
        food = await smart_food_search(q)
        if not food:
            unknown.append(name_ru or name_en or "неизвестно")
            continue

        c100 = float(food["calories"])
        kcal = c100 * grams / 100.0
        total_kcal += kcal

        enriched.append({
            "name": food["name"],
            "grams": grams,
            "kcal": kcal
        })

    await wait.delete()

    if not enriched:
        await state.clear()
        await message.answer(
            "❌ <b>Компоненты распознал, но не смог найти калории в базе.</b>\n\n"
            "Попробуйте другое фото.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📷 Попробовать снова", callback_data="add_food_photo")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
            ])
        )
        return

    await state.update_data(photo_items=enriched, photo_total=total_kcal)
    await state.set_state(PhotoStates.waiting_confirm)

    lines = ["📷 <b>Результат по фото (примерно):</b>\n"]
    for i, x in enumerate(enriched, 1):
        lines.append(f"{i}. <b>{x['name']}</b> — {x['grams']}г — {x['kcal']:.1f} ккал")
    lines.append(f"\n🔥 <b>Итого:</b> {total_kcal:.1f} ккал")
    lines.append(f"📌 <b>Уверенность ИИ:</b> {conf}")
    if notes:
        lines.append(f"\n<i>{notes}</i>")
    if unknown:
        lines.append("\n⚠️ <b>Не найдено в базе:</b> " + ", ".join(unknown[:10]))

    await message.answer("\n".join(lines), parse_mode="HTML", reply_markup=kb_photo_confirm())

@dp.message(PhotoStates.waiting_for_photo)
async def photo_not_photo(message: types.Message):
    await message.answer("📷 Пожалуйста, отправьте именно <b>фото</b> еды.", parse_mode="HTML")

@dp.callback_query(PhotoStates.waiting_confirm, F.data == "photo_confirm_add")
async def photo_confirm_add(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    data = await state.get_data()
    items = data.get("photo_items") or []
    total = float(data.get("photo_total", 0) or 0)

    uid = callback.from_user.id
    ud = get_user_data(uid)
    tm = now_local().strftime("%H:%M")

    for x in items:
        ud["foods"].append({"name": x["name"], "weight": int(x["grams"]), "calories": float(x["kcal"]), "time": tm})
        ud["total_calories"] += float(x["kcal"])

    await state.clear()
    await callback.message.edit_text(
        f"✅ <b>Добавлено из фото!</b>\n\n"
        f"🔥 Добавлено: {total:.1f} ккал\n"
        f"📊 Всего сегодня: {ud['total_calories']:.1f} ккал",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📷 Ещё фото", callback_data="add_food_photo")],
            [InlineKeyboardButton(text="📊 Итоги", callback_data="show_stats")],
            [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
        ])
    )

@dp.callback_query(PhotoStates.waiting_confirm, F.data == "photo_confirm_cancel")
async def photo_confirm_cancel(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.clear()
    await callback.message.edit_text("❌ Отменено.", parse_mode="HTML", reply_markup=kb_main())

# -------- STATS / DELETE ONE / CLEAR
@dp.callback_query(F.data == "show_stats")
async def show_stats(callback: types.CallbackQuery):
    await callback.answer()
    uid = callback.from_user.id
    ud = get_user_data(uid)

    foods = ud.get("foods", []) or []
    total = float(ud.get("total_calories") or 0)
    date_human = now_local().strftime("%d.%m.%Y")

    if not foods:
        await callback.message.edit_text(
            f"📊 <b>Итоги за {date_human}</b>\n\nНет записей.",
            parse_mode="HTML",
            reply_markup=kb_stats_menu(has_foods=False)
        )
        return

    lines = [f"📊 <b>Итоги за {date_human}:</b>\n"]
    for i, f in enumerate(foods, 1):
        lines.append(f"{i}. {f['name']} — {f['weight']}г ({f['calories']:.1f} ккал) в {f.get('time','')}".strip())
    lines.append(f"\n🔥 <b>Всего:</b> {total:.1f} ккал")

    await callback.message.edit_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=kb_stats_menu(has_foods=True)
    )

@dp.callback_query(F.data == "delete_menu")
async def delete_menu(callback: types.CallbackQuery):
    await callback.answer()
    uid = callback.from_user.id
    ud = get_user_data(uid)
    if not ud.get("foods"):
        await callback.message.edit_text("❌ Нет записей для удаления.", reply_markup=kb_stats_menu(False), parse_mode="HTML")
        return

    await callback.message.edit_text(
        "🗑️ <b>Выбери продукт для удаления:</b>",
        parse_mode="HTML",
        reply_markup=kb_delete_list(uid)
    )

@dp.callback_query(F.data.startswith("delete_one:"))
async def delete_one(callback: types.CallbackQuery):
    await callback.answer()
    uid = callback.from_user.id
    ud = get_user_data(uid)

    try:
        idx = int(callback.data.split(":")[1])
    except:
        await callback.message.edit_text("❌ Ошибка индекса.", parse_mode="HTML", reply_markup=kb_stats_menu(bool(ud.get("foods"))))
        return

    foods = ud.get("foods", []) or []
    if idx < 0 or idx >= len(foods):
        await callback.message.edit_text("❌ Запись не найдена.", parse_mode="HTML", reply_markup=kb_stats_menu(bool(foods)))
        return

    removed = foods.pop(idx)
    removed_kcal = float(removed.get("calories") or 0)
    ud["total_calories"] = max(0.0, float(ud.get("total_calories") or 0) - removed_kcal)

    await callback.message.edit_text(
        f"✅ <b>Удалено:</b> {removed.get('name','')}\n"
        f"🔥 {removed_kcal:.1f} ккал\n\n"
        f"📊 Теперь всего: {ud['total_calories']:.1f} ккал",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🗑️ Удалить ещё", callback_data="delete_menu")],
            [InlineKeyboardButton(text="📊 Итоги", callback_data="show_stats")],
            [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
        ])
    )

@dp.callback_query(F.data == "clear_day")
async def clear_day(callback: types.CallbackQuery):
    await callback.answer()
    uid = callback.from_user.id
    ud = get_user_data(uid)
    removed = float(ud.get("total_calories") or 0)

    user_data[uid] = {"foods": [], "total_calories": 0.0, "date": today_str()}

    await callback.message.edit_text(
        f"🗑️ <b>День очищен!</b>\n\nУдалено: {removed:.1f} ккал",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🍎 Добавить еду", callback_data="add_food")],
            [InlineKeyboardButton(text="📷 Еда по фото", callback_data="add_food_photo")],
            [InlineKeyboardButton(text="🏠 В меню", callback_data="main_menu")]
        ])
    )

# =========================
# IMPORTANT: CATCH-ALL ONLY TEXT, AND ONLY WHEN NO FSM STATE
# (чтобы фото НЕ ломалось)
# =========================
@dp.message(F.text)
async def catch_text(message: types.Message, state: FSMContext):
    if not message.text or message.text.startswith("/"):
        return

    if await state.get_state() is not None:
        return

    query = message.text.strip()
    wait = await message.answer("🔍 Ищу в базе...")
    food = await smart_food_search(query)
    await wait.delete()

    if not food:
        await message.answer(
            f"❌ Не нашёл: <b>{query}</b>\n\n"
            "Можешь предложить блюдо 👇",
            parse_mode="HTML",
            reply_markup=kb_not_found(query)
        )
        return

    await state.set_state(FoodStates.waiting_for_food_weight)
    await state.update_data(food_name=food["name"], calories_per_100=float(food["calories"]))

    text = "🌍 <b>Найдено:</b>\n" + format_food_info(food) + "Выберите вес:"
    await message.answer(text, parse_mode="HTML", reply_markup=kb_weights())

# =========================
# RUN
# =========================
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not BOT_TOKEN:
        print("❌ BOT_TOKEN пустой. Добавь в .env: BOT_TOKEN=...")
        return
    if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
        print("⚠️ Нет EDAMAM_APP_ID/EDAMAM_APP_KEY. Поиск по базе не будет работать.")
    if not OPENAI_API_KEY:
        print("⚠️ Нет OPENAI_API_KEY. Фото и умный фолбэк работать не будут.")
    if not BOT_TZ:
        print("⚠️ Не удалось загрузить timezone. Итоги в 21:00 будут по системному времени.")

    # запуск фоновой задачи 21:00
    asyncio.create_task(daily_summary_loop())

    print("✅ Bot starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
