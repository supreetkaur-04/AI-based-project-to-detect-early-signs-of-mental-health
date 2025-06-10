# alert.py

from telegram import Bot
import asyncio

BOT_TOKEN = "7952132172:AAGQkGyGggk3ypWp2nJYb8F2G1p3VF9oqg8"
COUNSELOR_CHAT_ID = "1714427341"

bot = Bot(token="7952132172:AAGQkGyGggk3ypWp2nJYb8F2G1p3VF9oqg8")
loop = asyncio.new_event_loop()  # Global event loop

async def send_alert(message):
    try:
        await bot.send_message(chat_id=COUNSELOR_CHAT_ID, text=message)
        print(f"✅ Alert sent to counselor: {message}")
    except Exception as e:
        print(f"❌ Failed to send alert: {e}")

# def trigger_alert(distress_message):
#     asyncio.run(send_alert(distress_message))


def trigger_alert(message):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_alert(message))

    
if __name__ == "__main__":
    print("Testing the alert system...")
    trigger_alert("⚠️ Test Alert: Distress detected! Please check on the individual.")
