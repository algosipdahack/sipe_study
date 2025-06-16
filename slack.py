from slack_sdk.webhook import WebhookClient
from dotenv import load_dotenv
import os

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
slack = WebhookClient(SLACK_WEBHOOK_URL)

slack.send(text="📢 슬랙 봇이 잘 작동합니다!")
