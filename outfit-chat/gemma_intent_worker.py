import argparse
import json
import logging
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LOGGER = logging.getLogger("gemma_intent_worker")
STYLE_BY_OCCASION = {
    "walk": "casual",
    "coffee": "casual",
    "brunch": "casual",
    "errands": "casual",
    "weekend": "casual",
    "trip": "casual",
    "travel": "casual",
    "vacation": "casual",
    "holiday": "casual",
    "flight": "casual",
    "work": "office",
    "office": "office",
    "meeting": "office",
    "interview": "formal",
    "conference": "formal",
    "theatre": "elegant",
    "theater": "elegant",
    "film": "elegant",
    "movie": "elegant",
    "cinema": "elegant",
    "party": "elegant",
    "dinner": "elegant",
    "date": "elegant",
    "wedding": "formal",
    "ceremony": "formal",
    "gym": "sport",
    "workout": "sport",
    "training": "sport",
    "run": "sport",
}
ALLOWED_STYLES = {"casual", "office", "sport", "elegant", "bohemian", "streetwear", "formal"}
SCOPE_KEYWORDS = ["outfit", "wear", "wearing", "look", "dress", "style", "wardrobe", "shuffle", "another option"]
SHUFFLE_KEYWORDS = ["shuffle", "another option", "another look", "different option", "different outfit"]
WEATHER_STATUS_KEYWORDS = ["sunny", "cloudy", "rainy", "windy", "snowy", "stormy"]
DATE_KEYWORDS = ["today", "tonight", "tomorrow", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CONSTRAINT_KEYWORDS = ["warmer", "cooler", "more casual", "less casual", "more formal", "less formal", "layered", "minimal", "sportier"]
LOCATION_STOP_WORDS = {
    "for",
    "a",
    "an",
    "the",
    "to",
    "with",
    "on",
    "at",
    "and",
    "but",
    "today",
    "tomorrow",
    "tonight",
    "this",
    "weekend",
    "weather",
    "casual",
    "office",
    "sport",
    "sporty",
    "elegant",
    "formal",
    "bohemian",
    "streetwear",
    "drink",
    "outfit",
    "look",
    "go",
    "going",
    "need",
    "want",
}

SYSTEM_PROMPT = """You extract outfit-planning intent for Stylora.
Return one JSON object only with these keys:
intent, is_in_scope, occasion_text, style_bucket, location, date_context, weather_summary, weather_status, temperature_c, constraints, shuffle_count, parser_source.
Rules:
- Stylora only handles outfit planning, wardrobe, shuffle, style adjustments, and weather-aware outfit generation.
- If the request is unrelated, set intent to "out_of_scope" and is_in_scope to false.
- style_bucket must be one of: casual, office, sport, elegant, bohemian, streetwear, formal, or null.
- Keep weather_summary/status/temperature_c null unless the user provided them or the message clearly states them.
- Keep location/date_context null unless the user explicitly gave them.
- constraints is a list of short phrases from the user's request.
- shuffle_count is the number of shuffle requests implied by the latest conversation.
- parser_source must be "gemma".
Do not add markdown or explanation."""


class GemmaIntentEngine:
    def __init__(self, model_id: str, max_new_tokens: int, temperature: float):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.processor = None
        self.model = None
        self.error = None
        self.mode = "initializing"
        self.ready = False
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            LOGGER.info("Loading Gemma model %s", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype="auto",
                device_map="auto"
            )
            self.mode = "gemma"
            self.ready = True
            LOGGER.info("Gemma model loaded.")
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = str(exc)
            self.mode = "heuristic"
            self.ready = True
            LOGGER.warning("Gemma load failed, using heuristic fallback: %s", exc)

    def parse(self, messages):
        if self.mode == "gemma" and self.processor is not None and self.model is not None:
            try:
                return self._parse_with_gemma(messages)
            except Exception as exc:  # pragma: no cover - environment dependent
                LOGGER.warning("Gemma parse failed, falling back to heuristic parsing: %s", exc)

        result = heuristic_parse(messages)
        result["parser_source"] = "heuristic"
        return result

    def _parse_with_gemma(self, messages):
        prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            prompt_messages.append({"role": message.get("role", "user"), "content": content})

        text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        payload = extract_json(decoded)
        if payload is None:
            raise ValueError("Gemma response did not contain valid JSON.")

        normalized = normalize_payload(payload)
        normalized["parser_source"] = "gemma"
        return normalized


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_payload(payload):
    style_bucket = payload.get("style_bucket")
    if isinstance(style_bucket, str):
        style_bucket = style_bucket.strip().lower()
        if style_bucket not in ALLOWED_STYLES:
            style_bucket = None
    else:
        style_bucket = None

    normalized = {
        "intent": payload.get("intent") or "generate_outfit",
        "is_in_scope": bool(payload.get("is_in_scope", True)),
        "occasion_text": payload.get("occasion_text"),
        "style_bucket": style_bucket,
        "location": payload.get("location"),
        "date_context": payload.get("date_context"),
        "weather_summary": payload.get("weather_summary"),
        "weather_status": payload.get("weather_status"),
        "temperature_c": payload.get("temperature_c"),
        "constraints": [value for value in payload.get("constraints", []) if isinstance(value, str) and value.strip()],
        "shuffle_count": int(payload.get("shuffle_count") or 0),
        "parser_source": payload.get("parser_source") or "gemma",
    }
    if not normalized["date_context"]:
        normalized["date_context"] = "today"
    return normalized


def heuristic_parse(messages):
    user_messages = [
        (message.get("content") or "").strip()
        for message in messages
        if (message.get("role") or "").lower() == "user" and (message.get("content") or "").strip()
    ]

    if not user_messages:
        return {
            "intent": "clarify_request",
            "is_in_scope": True,
            "occasion_text": None,
            "style_bucket": None,
            "location": None,
            "date_context": None,
            "weather_summary": None,
            "weather_status": None,
            "temperature_c": None,
            "constraints": [],
            "shuffle_count": 0,
            "parser_source": "heuristic",
        }

    full_text = " ".join(user_messages).lower()
    if not has_outfit_signals(" ".join(user_messages)):
        return {
            "intent": "out_of_scope",
            "is_in_scope": False,
            "occasion_text": None,
            "style_bucket": None,
            "location": None,
            "date_context": None,
            "weather_summary": None,
            "weather_status": None,
            "temperature_c": None,
            "constraints": [],
            "shuffle_count": 0,
            "parser_source": "heuristic",
        }

    result = {
        "intent": "generate_outfit",
        "is_in_scope": True,
        "occasion_text": None,
        "style_bucket": None,
        "location": None,
        "date_context": None,
        "weather_summary": None,
        "weather_status": None,
        "temperature_c": None,
        "constraints": [],
        "shuffle_count": 0,
        "parser_source": "heuristic",
    }

    for message in user_messages:
        lowered = message.lower()
        if any(keyword in lowered for keyword in SHUFFLE_KEYWORDS):
            result["shuffle_count"] += 1

        for style in ALLOWED_STYLES:
            if style in lowered:
                result["style_bucket"] = style
                result["occasion_text"] = result["occasion_text"] or style

        for occasion, mapped_style in STYLE_BY_OCCASION.items():
            if occasion in lowered:
                result["occasion_text"] = result["occasion_text"] or occasion
                result["style_bucket"] = result["style_bucket"] or mapped_style

        for keyword in CONSTRAINT_KEYWORDS:
            if keyword in lowered and keyword not in result["constraints"]:
                result["constraints"].append(keyword)

        weather = extract_weather(message)
        if weather is not None:
            result["weather_summary"] = weather["summary"]
            result["weather_status"] = weather["status"]
            result["temperature_c"] = weather["temperature_c"]

        location = extract_location(message)
        if location:
            result["location"] = location

        for keyword in DATE_KEYWORDS:
            if keyword in lowered:
                result["date_context"] = keyword
                break

    if not result["date_context"]:
        result["date_context"] = "today"

    result["intent"] = "shuffle_outfit" if result["shuffle_count"] > 0 else "generate_outfit"
    return result


def has_outfit_signals(message):
    lowered = message.lower()
    return (
        any(keyword in lowered for keyword in SCOPE_KEYWORDS)
        or any(style in lowered for style in ALLOWED_STYLES)
        or any(occasion in lowered for occasion in STYLE_BY_OCCASION)
        or any(keyword in lowered for keyword in WEATHER_STATUS_KEYWORDS)
        or any(keyword in lowered for keyword in DATE_KEYWORDS)
        or any(word in lowered for word in ["cold", "cool", "warm", "hot"])
        or bool(extract_location(message))
    )


def extract_weather(message):
    lowered = message.lower()
    status = next((keyword for keyword in WEATHER_STATUS_KEYWORDS if keyword in lowered), None)

    thermal_band = None
    if "freezing" in lowered:
        thermal_band = "freezing"
    elif "cold" in lowered or "chilly" in lowered:
        thermal_band = "cold"
    elif "cool" in lowered:
        thermal_band = "cool"
    elif "warm" in lowered:
        thermal_band = "warm"
    elif "hot" in lowered:
        thermal_band = "hot"

    temperature_c = None
    match = re.search(r"(-?\d{1,2}(?:\.\d+)?)\s*(?:°?\s*c|degrees?)", lowered)
    if match:
        temperature_c = float(match.group(1))
        thermal_band = thermal_band_from_temperature(temperature_c)

    if status is None and thermal_band is None:
        return None

    summary_parts = []
    if status:
        summary_parts.append(status)
    if temperature_c is not None:
        summary_parts.append(f"{temperature_c:g}C")
    elif thermal_band:
        summary_parts.append(thermal_band)

    return {
        "status": status or thermal_band or "mild",
        "temperature_c": temperature_c,
        "summary": ", ".join(summary_parts),
    }


def extract_location(message):
    contextual_candidate = extract_location_after_preposition(message)
    if contextual_candidate:
        return normalize_location_candidate(contextual_candidate)

    for segment in split_message_segments(message):
        standalone_match = re.search(r"^([A-Z][a-z]+(?:[\s-][A-Z][a-z]+){0,3})(?:\s+(?:today|tomorrow|tonight|this weekend|weekend))?$", segment.strip())
        if standalone_match:
            return normalize_location_candidate(standalone_match.group(1))

    return None


def normalize_location_candidate(value):
    normalized = value.strip()
    normalized = re.sub(r"\s+(today|tomorrow|tonight|this weekend|weekend)$", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def split_message_segments(message):
    return [segment.strip() for segment in re.split(r"[,;|]", message) if segment.strip()]


def extract_location_after_preposition(message):
    return (
        extract_location_after_keyword(message, "in")
        or extract_location_after_keyword(message, "to")
        or extract_location_after_keyword(message, "for")
    )


def extract_location_after_keyword(message, keyword):
    matches = re.finditer(
        rf"\b{keyword}\b\s+(.+?)(?=(?:\b(?:in|to|for)\b\s+)|$)",
        message,
        flags=re.IGNORECASE,
    )

    for match in reversed(list(matches)):
        tail = match.group(1).strip()
        if not tail:
            continue

        candidate_words = []
        for raw_word in tail.split():
            cleaned_word = raw_word.strip(",.!?;:")
            if not cleaned_word:
                continue
            if cleaned_word.lower() in LOCATION_STOP_WORDS:
                break
            if not re.match(r"^[A-Za-z-]+$", cleaned_word):
                break
            candidate_words.append(cleaned_word)
            if len(candidate_words) == 4:
                break

        if candidate_words:
            return " ".join(candidate_words)

    return None


def thermal_band_from_temperature(temperature_c):
    if temperature_c <= 4:
        return "freezing"
    if temperature_c <= 11:
        return "cold"
    if temperature_c <= 18:
        return "cool"
    if temperature_c <= 26:
        return "warm"
    return "hot"


class RequestHandler(BaseHTTPRequestHandler):
    engine = None

    def _json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ready": self.engine.ready, "mode": self.engine.mode, "error": self.engine.error})
            return

        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path != "/parse":
            self._json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            messages = payload.get("messages", [])
            result = self.engine.parse(messages)
            self._json(200, {"result": result})
        except Exception as exc:  # pragma: no cover - transport guard
            LOGGER.exception("Parse failed.")
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):  # pragma: no cover - keep stderr tidy
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-26B-A4B-it"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    RequestHandler.engine = GemmaIntentEngine(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), RequestHandler)
    LOGGER.info("Gemma intent worker listening on port %s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
