import base64
import os
import re
import time

import cv2
from pydantic import BaseModel, ValidationError

PROMPT = """You are analyzing consecutive frames from an indoor camera monitoring an elderly person near a bed.
The monitored person is marked with a GREEN box when the vision pipeline can see them. Other people, such as
caregivers, are not monitored: ignore them. The monitored person may be partly or fully under a blanket.
Answer using only what is visible. If the evidence is insufficient, use UNKNOWN.
Allowed states: LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, LYING_OUTSIDE_BED, UNKNOWN
Allowed locations: bed, floor, chair, standing_area, not_visible, unknown
Question: {question}

Respond with ONLY a JSON object with keys:
"state" (one allowed state), "location" (one allowed location), "confidence" (0 to 1), "reasoning" (one sentence)."""


class VLMAnswer(BaseModel):
    state: str
    location: str
    confidence: float
    reasoning: str


def encode_jpeg(img, max_side: int = 512) -> bytes:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.imencode(".jpg", small)[1].tobytes()


def to_data_url(img) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(img)).decode()


def parse_answer(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return VLMAnswer.model_validate_json(match.group(0))
    except ValidationError:
        return None


class BedrockVLM:
    name = "bedrock-openai"

    def __init__(self, model: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"], base_url=os.environ["OPENAI_BASE_URL"],
                             project=os.getenv("OPENAI_PROJECT_ID") or None)

    def ask(self, images, question: str):
        from openai import RateLimitError
        content = [{"type": "image_url", "image_url": {"url": to_data_url(img)}} for img in images]
        content.append({"type": "text", "text": PROMPT.format(question=question)})
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": content}],
                    temperature=0, max_tokens=400)
                return parse_answer(response.choices[0].message.content)
            except RateLimitError:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                print("VLM call failed: rate limited")
                return None
            except Exception as exc:
                print(f"VLM call failed: {exc}")
                return None


def make_vlm():
    missing = [v for v in ("AWS_BEARER_TOKEN_BEDROCK", "OPENAI_BASE_URL", "VLM_MODEL") if not os.getenv(v)]
    if missing:
        print(f"VLM disabled: set {', '.join(missing)} in .env")
        return None
    try:
        return BedrockVLM(os.environ["VLM_MODEL"])
    except Exception as exc:
        print(f"VLM disabled: {exc}")
        return None
