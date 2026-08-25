# ═══════════════════════════════════════════════════════════
#  🚀 SIDOX SIGNAL BOT WITH MULTI-AI FAILOVER (8 KEYS)
#  Version: Enterprise Final
# ═══════════════════════════════════════════════════════════

import os
import json
import time
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot, Update
from keep_alive import keep_alive
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --- AI PROVIDERS ---
import google.generativeai as genai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# --- ARABIC CONSOLE FIX ---
import arabic_reshaper
from bidi.algorithm import get_display

def ar(text: str) -> str:
    """إصلاح عرض اللغة العربية في الشاشة السوداء (RTL)"""
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)

# ═══════════════════════════════════════════
# 1. الإعدادات و جلب المفاتيح (8 Keys)
# ═══════════════════════════════════════════
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID')

# جلب مصفوفة المفاتيح من ملف .env
GEMINI_KEYS = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")]
CLAUDE_KEYS = [os.getenv("ANTHROPIC_API_KEY_1"), os.getenv("ANTHROPIC_API_KEY_2"), os.getenv("ANTHROPIC_API_KEY_3")]
OPENAI_KEYS = [os.getenv("OPENAI_API_KEY_1"), os.getenv("OPENAI_API_KEY_2"), os.getenv("OPENAI_API_KEY_3")]

# تنظيف المفاتيح الفارغة
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]
CLAUDE_KEYS = [k for k in CLAUDE_KEYS if k]
OPENAI_KEYS = [k for k in OPENAI_KEYS if k]

if not BOT_TOKEN or not CHAT_ID or not GEMINI_KEYS:
    print(ar("❌ مفاتيح أساسية ناقصة في ملف .env (تأكد من وجود توكن، أيدي، ومفتاح جيميناي واحد على الأقل)"))
    input(ar("اضغط Enter للخروج..."))
    exit()

# ─── اكتشاف أفضل موديل لـ Gemini مرة واحدة عند البدء ───
genai.configure(api_key=GEMINI_KEYS[0])
def detect_gemini_model():
    priority = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    try:
        available = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for name in priority:
            if name in available: return name
        for name in available:
            if "flash" in name: return name
        if available: return available[0]
    except Exception as e:
        print(ar(f"[GEMINI] Detection failed: {e}"))
    return "gemini-3.6-flash"

BEST_GEMINI_MODEL = detect_gemini_model()
print(ar(f"[SYSTEME] ✅ Gemini Model Selected: {BEST_GEMINI_MODEL}"))

# ─── إعداد Binance و المتغيرات ───
exchange = ccxt.bybit({'enableRateLimit': True})

os.makedirs("trades", exist_ok=True)
TRADES_FILE = "trades/trades_state.json"

SCAN_INTERVAL   = 300      # مسح كل 5 دقائق
TRACK_INTERVAL  = 20       # متابعة الصفقات كل 20 ثانية
MIN_CONFIDENCE  = 70       # أقل نسبة ثقة لقبول إشارة

# ═══════════════════════════════════════════
# 2. أدوات مساعدة
# ═══════════════════════════════════════════
def is_authorized(chat_id) -> bool:
    return str(chat_id) == str(CHAT_ID)

async def send_message(text: str):
    try:
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='HTML')
    except Exception as e:
        print(ar(f"[TELEGRAM] خطأ إرسال: {e}"))

def load_trades() -> dict:
    if not os.path.exists(TRADES_FILE): return {}
    try:
        with open(TRADES_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return {}

def save_trades(trades: dict):
    with open(TRADES_FILE, 'w', encoding='utf-8') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════
# 3. التحليل الفني (Binance)
# ═══════════════════════════════════════════
def get_technical_data(symbol: str) -> dict:
    try:
        if '/' not in symbol: symbol = symbol.upper() + '/USDT'
        ticker     = exchange.fetch_ticker(symbol)
        price      = ticker['last']
        change_24h = ticker.get('percentage', 0) or 0
        volume_24h = ticker.get('quoteVolume', 0) or 0
        high_24h   = ticker.get('high', 0) or 0
        low_24h    = ticker.get('low', 0) or 0

        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_1h  = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h',  limit=200), columns=cols)
        df_4h  = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h',  limit=100), columns=cols)
        df_15m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=100), columns=cols)

        # ─── مؤشرات 1H ───
        df_1h['ema20']  = df_1h['close'].ewm(span=20,  adjust=False).mean()
        df_1h['ema50']  = df_1h['close'].ewm(span=50,  adjust=False).mean()
        df_1h['ema200'] = df_1h['close'].ewm(span=200, adjust=False).mean()

        delta = df_1h['close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        df_1h['rsi'] = 100 - (100 / (1 + gain / loss))

        ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()
        df_1h['macd']      = ema12 - ema26
        df_1h['signal']    = df_1h['macd'].ewm(span=9, adjust=False).mean()
        df_1h['macd_hist'] = df_1h['macd'] - df_1h['signal']

        df_1h['tr'] = np.maximum(df_1h['high'] - df_1h['low'], 
                      np.maximum(abs(df_1h['high'] - df_1h['close'].shift()), 
                                 abs(df_1h['low'] - df_1h['close'].shift())))
        df_1h['atr'] = df_1h['tr'].rolling(14).mean()

        df_1h['bb_mid'] = df_1h['close'].rolling(20).mean()
        bb_std = df_1h['close'].rolling(20).std()
        df_1h['bb_upper'] = df_1h['bb_mid'] + 2 * bb_std
        df_1h['bb_lower'] = df_1h['bb_mid'] - 2 * bb_std

        df_1h['vol_sma20'] = df_1h['volume'].rolling(20).mean()
        rel_volume = df_1h['volume'].iloc[-1] / df_1h['vol_sma20'].iloc[-1]

        # ─── مؤشرات 4H & 15M ───
        df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
        df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
        d4 = df_4h['close'].diff()
        df_4h['rsi'] = 100 - (100 / (1 + d4.where(d4 > 0, 0).rolling(14).mean() / -d4.where(d4 < 0, 0).rolling(14).mean()))

        d15 = df_15m['close'].diff()
        df_15m['rsi'] = 100 - (100 / (1 + d15.where(d15 > 0, 0).rolling(14).mean() / -d15.where(d15 < 0, 0).rolling(14).mean()))

        close_4h, ema20_4h, ema50_4h = df_4h['close'].iloc[-1], df_4h['ema20'].iloc[-1], df_4h['ema50'].iloc[-1]
        trend_4h = "BULLISH" if close_4h > ema20_4h > ema50_4h else "BEARISH" if close_4h < ema20_4h < ema50_4h else "NEUTRAL"

        return {
            'success': True, 'symbol': symbol, 'price': price, 'change_24h': change_24h,
            'volume_24h': volume_24h, 'high_24h': high_24h, 'low_24h': low_24h,
            'ema20_1h': df_1h['ema20'].iloc[-1], 'ema50_1h': df_1h['ema50'].iloc[-1], 'ema200_1h': df_1h['ema200'].iloc[-1],
            'rsi_1h': df_1h['rsi'].iloc[-1], 'macd': df_1h['macd'].iloc[-1], 'macd_signal': df_1h['signal'].iloc[-1],
            'macd_hist': df_1h['macd_hist'].iloc[-1], 'atr': df_1h['atr'].iloc[-1], 'atr_pct': (df_1h['atr'].iloc[-1] / price) * 100,
            'bb_upper': df_1h['bb_upper'].iloc[-1], 'bb_mid': df_1h['bb_mid'].iloc[-1], 'bb_lower': df_1h['bb_lower'].iloc[-1],
            'relative_volume': rel_volume, 'swing_high': df_1h['high'].rolling(20).max().iloc[-1],
            'swing_low': df_1h['low'].rolling(20).min().iloc[-1], 'trend_4h': trend_4h,
            'rsi_4h': df_4h['rsi'].iloc[-1], 'rsi_15m': df_15m['rsi'].iloc[-1],
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ═══════════════════════════════════════════
# 4. MULTI-KEY AI ROTATION & FAILOVER SYSTEM
# ═══════════════════════════════════════════
def extract_json(raw_text: str) -> dict:
    """Helper to safely extract JSON from LLM responses."""
    try:
        clean = raw_text.strip()
        if '```json' in clean: clean = clean.split('```json')[1].split('```')[0].strip()
        elif '```' in clean: clean = clean.split('```')[1].split('```')[0].strip()
        return {'success': True, 'data': json.loads(clean)}
    except Exception as e:
        return {'success': False, 'error': f"JSON parsing failed: {e}"}

async def ask_ai_with_failover(d: dict) -> dict:
    prompt = f"""
أنت محلل تداول احترافي في العملات الرقمية. حلّل البيانات الحقيقية التالية لعملة {d['symbol']}:

── البيانات الأساسية ──
السعر: {d['price']:.6f} USDT | تغير 24س: {d['change_24h']:.2f}%
حجم 24س: {d['volume_24h']:,.0f} USDT

── المؤشرات (1H) ──
الاتجاه 4H: {d['trend_4h']}
RSI 1H: {d['rsi_1h']:.2f} | RSI 4H: {d['rsi_4h']:.2f} | RSI 15M: {d['rsi_15m']:.2f}
EMA20: {d['ema20_1h']:.6f} | EMA50: {d['ema50_1h']:.6f} | EMA200: {d['ema200_1h']:.6f}
MACD: {d['macd']:.6f} | Signal: {d['macd_signal']:.6f} | Hist: {d['macd_hist']:.6f}
ATR: {d['atr']:.6f} ({d['atr_pct']:.2f}%)
الحجم النسبي: {d['relative_volume']:.2f}x

── المطلوب ──
1. حدد الإشارة: LONG أو SHORT أو WAIT
2. إذا LONG/SHORT: أعطِ Entry, Stop Loss, TP1, TP2, TP3
3. R:R يجب ألا يقل عن 1:2
4. المدة المتوقعة للصفقة

أجب بصيغة JSON فقط بدون أي نص إضافي:
{{
  "signal": "LONG", "confidence": 75, "entry": 0.0, "stop_loss": 0.0,
  "tp1": 0.0, "tp2": 0.0, "tp3": 0.0, "duration": "12-24 ساعة"
}}
"""
    # ── PRIMARY: GEMINI KEYS ──
    for i, key in enumerate(GEMINI_KEYS):
        try:
            print(ar(f"  🧠 [Gemini Key {i+1}] جاري التحليل..."))
            genai.configure(api_key=key)
            model = genai.GenerativeModel(BEST_GEMINI_MODEL)
            # Use to_thread to prevent blocking the async loop
            response = await asyncio.to_thread(model.generate_content, prompt)
            return extract_json(response.text)
        except Exception as e:
            print(ar(f"  ⚠️ [Gemini Key {i+1}] فشل (تجاوز الحصة/خطأ): {e}"))

    # ── BACKUP 1: CLAUDE KEYS ──
    for i, key in enumerate(CLAUDE_KEYS):
        try:
            print(ar(f"  🤖 [Claude Key {i+1}] جاري التحليل كبديل..."))
            client = AsyncAnthropic(api_key=key)
            response = await client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return extract_json(response.content[0].text)
        except Exception as e:
            print(ar(f"  ⚠️ [Claude Key {i+1}] فشل: {e}"))

    # ── BACKUP 2: OPENAI KEYS ──
    for i, key in enumerate(OPENAI_KEYS):
        try:
            print(ar(f"  👁️ [OpenAI Key {i+1}] جاري التحليل كبديل أخير..."))
            client = AsyncOpenAI(api_key=key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            print(ar(f"  ⚠️ [OpenAI Key {i+1}] فشل: {e}"))

    return {'success': False, 'error': 'All 8 API keys exhausted/failed.'}

# ═══════════════════════════════════════════
# 5. تنسيق الرسالة و حفظ الصفقات (SIDOX-TREAD)
# ═══════════════════════════════════════════
def format_signal_message(t: dict, ai: dict) -> str:
    signal = ai.get('signal', 'WAIT')
    confidence = ai.get('confidence', 0)
    icons = {'LONG': ("🟢", "LONG (شراء)"), 'SHORT': ("🔴", "SHORT (بيع)"), 'WAIT': ("⚪", "WAIT (انتظار)")}
    s_icon, s_text = icons.get(signal, icons['WAIT'])

    msg = (
        f"---------------------\n"
        f"🪙 <b>{t['symbol']}</b>\n"
        f"💰 السعر: ${t['price']:,.6f}\n"
        f"📈 24h: {t['change_24h']:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{s_icon} <b>الإشارة: {s_text}</b>\n"
        f"✅ الثقة: {confidence}/100\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
    if signal in ('LONG', 'SHORT') and ai.get('entry') and ai.get('stop_loss'):
        msg += (
            f"\n💰 الدخول: {ai['entry']:,.6f}\n🛑 Stop Loss: {ai['stop_loss']:,.6f}\n"
            f"-------------------------------\n🎯 الأهداف:\n"
            f"TP1: {ai['tp1']:,.6f}\nTP2: {ai['tp2']:,.6f}\nTP3: {ai['tp3']:,.6f}\n"
            f"-------------------------------\n⏱️ المدة المتوقعة: {ai.get('duration', 'غير محدد')}\n"
            f"-------------------------------\nSIDOX-TREAD\n-------------------------------"
        )
    else:
        msg += "\n-------------------------------\nSIDOX-TREAD\n-------------------------------"
    return msg

def register_trade(tech: dict, ai: dict) -> str | None:
    if not all([ai.get('entry'), ai.get('stop_loss'), ai.get('tp1'), ai.get('tp2'), ai.get('tp3')]): return None
    trades = load_trades()
    trade_id = tech['symbol'].replace('/', '') + "-" + datetime.now().strftime('%Y%m%d%H%M%S')
    trades[trade_id] = {
        "symbol": tech['symbol'], "direction": ai.get('signal'), "status": "ACTIVE",
        "entry": ai['entry'], "sl": ai['stop_loss'], "tp1": ai['tp1'], "tp2": ai['tp2'], "tp3": ai['tp3'],
        "tp1_status": "NOT_HIT", "tp2_status": "NOT_HIT", "tp3_status": "NOT_HIT",
        "sl_moved_to_entry": False, "duration": ai.get('duration', '—'),
        "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_trades(trades)
    return trade_id

# ═══════════════════════════════════════════
# 6. الماسح التلقائي المفلتر
# ═══════════════════════════════════════════
async def market_scanner():
    print(ar(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📡 بدء مسح السوق والتصفية..."))
    try:
        tickers = exchange.fetch_tickers()
        
        # ─── فلتر استخراج العملات ───
        usdt_pairs = [
            {'symbol': s, 'volume': d['quoteVolume'], 'change': d.get('percentage', 0) or 0} 
            for s, d in tickers.items() 
            if s.endswith('/USDT') 
            and d.get('quoteVolume', 0) > 5000000 
            and d.get('last', 0) < 1000  # <--- الفلتر السحري: يستثني أي عملة سعرها فوق 1000 دولار
        ]
        
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        active_coins = [p for p in usdt_pairs[:100] if p['change'] > 1.0][:30]

        trades, active_symbols = load_trades(), {d['symbol'] for d in load_trades().values() if d.get('status') == 'ACTIVE'}
        candidates = []

        for coin in active_coins:
            if coin['symbol'] in active_symbols: continue
            tech = get_technical_data(coin['symbol'])
            if not tech['success']: continue

            score = 0
            if tech['price'] > tech['ema20_1h'] > tech['ema50_1h'] > tech['ema200_1h']: score += 30
            elif tech['price'] > tech['ema50_1h']: score += 10
            
            if 50 <= tech['rsi_1h'] <= 65: score += 20
            elif 65 < tech['rsi_1h'] <= 70: score += 10
            
            if tech['macd'] > tech['macd_signal'] and tech['macd_hist'] > 0: score += 20
            if tech['relative_volume'] >= 1.5: score += 15
            elif tech['relative_volume'] >= 1.0: score += 5
            if tech['atr_pct'] >= 2.0: score += 15

            if score >= 65: candidates.append({'tech': tech, 'score': score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:3]

        if not top_candidates: return
        print(ar(f"[SCANNER] التصفية اكتملت. سيتم إرسال أفضل {len(top_candidates)} للذكاء الاصطناعي."))

        sent = 0
        for c in top_candidates:
            sym, score = c['tech']['symbol'], c['score']
            print(ar(f"\n  🔍 طلب تحليل لـ {sym} (نقاط: {score}/100)"))
            
            # --- THE AI FAILOVER PIPELINE CALL ---
            res = await ask_ai_with_failover(c['tech'])

            if not res['success']:
                print(ar(f"  ❌ فشل كلي للذكاء الاصطناعي: {res['error']}"))
                await asyncio.sleep(5)
                continue

            ai = res['data']
            if ai.get('signal') in ('LONG', 'SHORT') and ai.get('confidence', 0) >= MIN_CONFIDENCE:
                msg = "-------------------\n🚨 <b>إشارة تلقائية جديدة</b>\n" + format_signal_message(c['tech'], ai)
                await send_message(msg)
                register_trade(c['tech'], ai)
                sent += 1
                print(ar(f"  ✅ إشارة مُرسلة: {sym} | {ai['signal']} | الثقة: {ai['confidence']}%"))
            else:
                print(ar(f"  ⚪ {sym}: تم التجاهل (الإشارة: {ai.get('signal')}, الثقة: {ai.get('confidence')}%)"))

            await asyncio.sleep(5) # Delay to prevent Binance/Local rate limits

        print(ar(f"[SCANNER] ✅ انتهت الدورة - {sent} إشارات."))
    except Exception as e:
        print(ar(f"[SCANNER] ❌ خطأ عام: {e}"))

# ═══════════════════════════════════════════
# 7. متابع الصفقات (TRADE TRACKER)
# ═══════════════════════════════════════════
async def trade_tracker():
    trades = load_trades()
    if not trades: return
    updated = False

    for tid, d in trades.items():
        if d.get('status') != 'ACTIVE': continue
        try: price = exchange.fetch_ticker(d['symbol'])['last']
        except: continue

        is_long = d.get('direction', 'LONG') == 'LONG'
        entry, sl, tp1, tp2, tp3 = d['entry'], d['sl'], d['tp1'], d['tp2'], d['tp3']
        hit = lambda t: price >= t if is_long else price <= t
        sl_hit = lambda: price <= sl if is_long else price >= sl

        if d['tp1_status'] == 'NOT_HIT' and hit(tp1):
            d['tp1_status'], d['sl'], d['sl_moved_to_entry'], updated = 'HIT', entry, True, True
            await send_message(f"🟢 <b>TP1 تحقق!</b>\n\n🪙 {d['symbol']}\n🎯 TP1: {tp1:,.6f}\n🔒 SL نُقل للدخول: {entry:,.6f}")
        elif d['tp1_status'] == 'HIT' and d['tp2_status'] == 'NOT_HIT' and hit(tp2):
            d['tp2_status'], updated = 'HIT', True
            await send_message(f"🟢 <b>TP2 تحقق!</b>\n\n🪙 {d['symbol']}\n🎯 TP2: {tp2:,.6f}\n🎯 الهدف الأخير: TP3 ({tp3:,.6f})")
        elif d['tp2_status'] == 'HIT' and d['tp3_status'] == 'NOT_HIT' and hit(tp3):
            d['tp3_status'], d['status'], updated = 'HIT', 'COMPLETED', True
            await send_message(f"🏆 <b>الصفقة مكتملة!</b>\n\n🪙 {d['symbol']}\n🎯 TP3: {tp3:,.6f}\n✅ تم إغلاق الصفقة بنجاح.")
        elif sl_hit():
            d['status'], updated = 'CLOSED_SL', True
            emoji, note = ("🛡️", "خروج عند الدخول (بدون خسارة)") if d.get('sl_moved_to_entry') else ("🔴", "خسارة")
            await send_message(f"{emoji} <b>Stop Loss</b>\n\n🪙 {d['symbol']}\n❌ الخروج: {sl:,.6f}\n📉 {note}")

    if updated: save_trades(trades)

# ═══════════════════════════════════════════
# 8. أوامر تيليغرام
# ═══════════════════════════════════════════
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id): return
    msg = (
        "👋 <b>أهلاً بك في بوت SIDOX-TREAD!</b> 🚀\n\n"
        "⚙️ <b>كيف أعمل؟</b>\n"
        "1️⃣ أراقب أفضل 100 عملة.\n"
        "2️⃣ أستخدم 8 مفاتيح ذكاء اصطناعي (Gemini/Claude/OpenAI).\n"
        "3️⃣ أرسل إشارات (دخول، أهداف، SL).\n\n"
        "📝 <b>الأوامر:</b>\n"
        "• اكتب رمز أي عملة (<code>BTC</code>)\n"
        "• <code>/top100</code> مسح فوري\n"
        "• <code>/status</code> الصفقات النشطة\n\n"
        "⚠️ <b>تنويه:</b> قراراتك مسؤوليتك الشخصية."
    )
    await update.message.reply_text(msg, parse_mode='HTML')

async def handle_coin_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id): return
    text = update.message.text.strip().upper()
    print(ar(f"\n📩 طلب تحليل يدوي: {text}"))

    w = await update.message.reply_text(f"🔍 جاري تحليل {text}...\n⏳ انتظر...")
    
    tech = get_technical_data(text)
    if not tech['success']:
        return await w.edit_text(f"❌ خطأ من المنصة لعملة {text}:\n<code>{tech.get('error')}</code>", parse_mode='HTML')

    await w.edit_text("✅ جلب البيانات تم\n🧠 جاري إرسالها للذكاء الاصطناعي...")
    
    # AI FAILOVER PIPELINE
    res = await ask_ai_with_failover(tech)
    
    if not res['success']:
        return await w.edit_text(f"❌ خطأ شامل في كل المفاتيح:\n<code>{res['error'][:300]}</code>", parse_mode='HTML')

    ai = res['data']
    await w.edit_text(format_signal_message(tech, ai), parse_mode='HTML')
    if ai.get('signal') in ('LONG', 'SHORT') and ai.get('confidence', 0) >= MIN_CONFIDENCE:
        tid = register_trade(tech, ai)
        if tid: await update.message.reply_text(f"✅ تمت الإضافة للمتابعة\n🆔 <code>{tid}</code>", parse_mode='HTML')
async def scan_top100_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id): return
    w = await update.message.reply_text("⏳ جاري مسح السوق...\n🧠 سيتم تصفية أفضل 3 عملات للذكاء الاصطناعي.")
    await market_scanner()
    await w.edit_text("✅ اكتمل المسح.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id): return
    active = {k: v for k, v in load_trades().items() if v.get('status') == 'ACTIVE'}
    msg = f"📊 <b>حالة النظام</b>\n🟢 نشطة: {len(active)}\n🕒 {datetime.now().strftime('%H:%M:%S')}"
    for tid, d in active.items():
        msg += f"\n\n🪙 <b>{d['symbol']}</b> | {d['direction']}\n💰 دخول: {d['entry']:,.6f}\n🛑 SL: {d['sl']:,.6f}"
    await update.message.reply_text(msg, parse_mode='HTML')

# ═══════════════════════════════════════════
# 9. تشغيل النظام والمهام الخلفية
# ═══════════════════════════════════════════
async def background_worker():
    print(ar("[SYSTEME] ✅ المهام الخلفية بدأت (التصفية + التناوب الذكي)"))
    await asyncio.sleep(10)
    last_scan = 0
    while True:
        try:
            now = time.time()
            if now - last_scan >= SCAN_INTERVAL:
                last_scan = now
                await market_scanner()
            await trade_tracker()
            await asyncio.sleep(TRACK_INTERVAL)
        except asyncio.CancelledError: break
        except Exception as e:
            print(ar(f"[BACKGROUND] خطأ: {e}"))
            await asyncio.sleep(10)

async def on_startup(app: Application):
    asyncio.create_task(background_worker())
    try:
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text=f"🤖 <b>تم تشغيل SIDOX-TREAD</b>\n\n🔑 المفاتيح النشطة: 8\n📡 المسح الآلي: مفعل\n👁 المتابعة: مفعلة",
            parse_mode='HTML'
        )
    except Exception: pass

def main():
    print(ar("=" * 55))
    print(ar(" 🚀 SIDOX-TREAD: MULTI-KEY ENTERPRISE SYSTEM"))
    print(ar("=" * 55))
    
    app = Application.builder().token(BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(CommandHandler("start",   start_command))
    app.add_handler(CommandHandler("top100",  scan_top100_command))
    app.add_handler(CommandHandler("status",  status_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_request))
    
    print(ar("[SYSTEME] ✅ في انتظار الأوامر..."))
    
    # --- التعديل السحري لحل مشكلة Event Loop في Render ---
    app.run_polling(drop_pending_updates=True, close_loop=False)

if __name__ == "__main__":
    from keep_alive import keep_alive
    keep_alive()  # إبقاء السيرفر مستيقظاً
    
    # إنشاء Event Loop يدوياً لحل مشكلة Python 3.14+
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        main()
    except KeyboardInterrupt:
        pass
