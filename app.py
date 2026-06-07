import base64
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "google/gemma-4-31b-it:free"

PROMPT = """당신은 식물 전문가입니다. 사진 속 식물을 분석하고 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

{
  "name_korean": "한국어 식물 이름",
  "name_scientific": "학명",
  "watering": "물 주기 빈도 (예: 주 1~2회)",
  "sunlight": "필요 햇빛량 (예: 밝은 간접광)",
  "difficulty": "쉬움",
  "tips": "핵심 관리 팁을 1~2문장으로"
}

difficulty 값은 반드시 '쉬움', '보통', '어려움' 중 하나여야 합니다.
사진에 식물이 없거나 인식 불가 시: {"error": "식물을 인식할 수 없습니다."}"""


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files["image"]
    mime = file.content_type or "image/jpeg"
    b64 = base64.b64encode(file.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return jsonify(json.loads(raw.strip()))

    except json.JSONDecodeError:
        return jsonify({"error": "응답 파싱 실패. 다시 시도해주세요."}), 500
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            return jsonify({"error": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."}), 429
        return jsonify({"error": msg}), 500


if __name__ == "__main__":
    app.run()
