from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import random
import unicodedata

app = Flask(__name__)
CORS(app)

# Load model
model_name = "vinai/bartpho-word"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)

# Load and validate training data
try:
    with open("chatbot_training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    if not isinstance(training_data.get("intents"), list):
        raise ValueError("JSON file must contain an 'intents' list")
    print("JSON file loaded successfully. Patterns for 'inquire_product':", 
          [p for intent in training_data["intents"] if intent["intent"] == "inquire_product" for p in intent["examples"]])
except Exception as e:
    print(f"Error loading JSON: {e}")
    training_data = {"intents": [], "products": []}

# ==================== INTENT DETECTION ====================
def detect_intent(user_input):
    user_input_normalized = unicodedata.normalize("NFKC", user_input.lower().strip())
    for intent in training_data.get("intents", []):
        for pattern in intent.get("examples", []):
            pattern_normalized = unicodedata.normalize("NFKC", pattern.lower().strip())
            print(f"Checking pattern: '{pattern_normalized}' against input: '{user_input_normalized}'")  # Debug log
            if pattern_normalized == user_input_normalized or pattern_normalized in user_input_normalized:
                print(f"Matched intent: {intent['intent']} with pattern: '{pattern}'")  # Debug log
                price_max, color, category, pet_type, size, material, location = extract_query_info(user_input)
                response = random.choice(intent["responses"])
                response = response.replace("{clothing_type}", category or "quần áo")
                response = response.replace("{pet_type}", pet_type or "thú cưng")
                response = response.replace("{size}", size or "phù hợp")
                response = response.replace("{color}", color or "đẹp")
                response = response.replace("{location}", location or "bạn")
                response = response.replace("{age}", "phù hợp")
                return response
    print(f"No intent matched for input: '{user_input_normalized}'")  # Debug log
    return None

# ==================== PRODUCT FILTERING ====================
def recommend_products(price_max=None, color=None, category=None, pet_type=None, size=None, material=None):
    products = training_data.get("products", [])
    results = []
    for product in products:
        if (price_max is None or product["price"] <= price_max) and \
           (color is None or product["color"].lower() == color.lower() if color else True) and \
           (category is None or product["name"].lower().find(category.lower()) != -1 if category else True) and \
           (pet_type is None or product["pet_type"].lower() == pet_type.lower() if pet_type else True) and \
           (size is None or product["size"].lower() == size.lower() if size else True) and \
           (material is None or product["material"].lower() == material.lower() if material else True):
            results.append(product)
    return results

# ==================== INFO EXTRACTION ====================
def extract_query_info(user_input):
    price_max = color = category = pet_type = size = material = location = None
    user_input_lower = unicodedata.normalize("NFKC", user_input.lower().strip())

    if "dưới" in user_input_lower:
        try:
            price_max = int(user_input_lower.split("dưới")[-1].split("k")[0].strip()) * 1000
        except:
            pass

    if "màu" in user_input_lower:
        color_words = user_input_lower.split("màu")[-1].strip().split()
        if color_words:
            color = color_words[0]

    if "áo" in user_input_lower:
        category = "áo"
    elif "váy" in user_input_lower:
        category = "váy"
    elif "quần" in user_input_lower:
        category = "quần"
    elif "yếm" in user_input_lower:
        category = "yếm"

    if "chó" in user_input_lower:
        pet_type = "chó"
    elif "mèo" in user_input_lower:
        pet_type = "mèo"

    if "size s" in user_input_lower or " s " in user_input_lower:
        size = "S"
    elif "size m" in user_input_lower or " m " in user_input_lower:
        size = "M"
    elif "size l" in user_input_lower or " l " in user_input_lower:
        size = "L"

    if "cotton" in user_input_lower:
        material = "cotton"
    elif "voan" in user_input_lower:
        material = "voan"
    elif "jeans" in user_input_lower:
        material = "jeans"
    elif "len" in user_input_lower:
        material = "len"

    if "hà nội" in user_input_lower:
        location = "Hà Nội"
    elif "tp.hcm" in user_input_lower or "sài gòn" in user_input_lower:
        location = "TP.HCM"
    elif "đà nẵng" in user_input_lower:
        location = "Đà Nẵng"
    elif "cần thơ" in user_input_lower:
        location = "Cần Thơ"

    return price_max, color, category, pet_type, size, material, location

# ==================== RESPONSE GENERATION ====================
def generate_response(user_input):
    user_input_normalized = unicodedata.normalize("NFKC", user_input.strip())
    user_input_lower = user_input_normalized.lower()

    # Handle vague or short inputs
    if len(user_input_normalized) <= 3 or user_input_lower in ["có", "ok", "ừ", "vâng"]:
        return "Dạ, bạn muốn tìm sản phẩm nào cho bé nhà mình nhỉ? Mình có áo, váy, quần cho chó và mèo, giá từ 150k-300k! 😊"

    # Prioritize intent matching
    intent_response = detect_intent(user_input_normalized)
    if intent_response:
        return intent_response

    # Handle specific keywords
    if user_input_lower in ["hi", "chào", "hello", "xin chào"]:
        return "Chào bạn! Mình là trợ lý tư vấn quần áo thú cưng đây. Bạn muốn tìm sản phẩm nào cho bé nhà mình nhỉ? 😊"

    if user_input_lower in ["cảm ơn", "thank you", "cám ơn"]:
        return "Không có gì đâu bạn! Nếu cần thêm gì, cứ nói với mình nhé! 😄"

    price_max, color, category, pet_type, size, material, location = extract_query_info(user_input_normalized)

    if any(keyword in user_input_lower for keyword in ["giặt", "bảo quản", "phơi"]):
        prompt = f"Hướng dẫn ngắn gọn cách bảo quản {category or 'quần áo thú cưng'}, trả lời tự nhiên như nhân viên bán hàng."
        full_response = generator(prompt, max_new_tokens=150, truncation=True, num_return_sequences=1)[0]['generated_text']
        response = full_response.replace(prompt, "").strip()
        if response and response[-1] not in ".!?":
            response += "."
        return response or "Nên giặt tay với nước mát, tránh chất tẩy mạnh và phơi nơi thoáng mát để giữ form quần áo nhé! 😊"

    if any(keyword in user_input_lower for keyword in ["giao hàng", "bao lâu", "phí ship", "khi nào tới"]):
        prompt = f"Thông báo thời gian giao hàng và phí ship{(f' cho {location}' if location else '')}, trả lời tự nhiên như nhân viên bán hàng."
        full_response = generator(prompt, max_new_tokens=150, truncation=True, num_return_sequences=1)[0]['generated_text']
        response = full_response.replace(prompt, "").strip()
        if response and response[-1] not in ".!?":
            response += "."
        return response or f"Bạn ở {location or 'khu vực của bạn'} thì hàng sẽ tới trong 1-2 ngày, phí ship 30k, miễn phí cho đơn từ 500k nha! 😊"

    products = recommend_products(price_max, color, category, pet_type, size, material)
    if products:
        product_list = ", ".join([f"{p['name']} (Giá: {p['price']} VNĐ, Màu: {p['color']})" for p in products])
        return f"Dạ, shop có {product_list}. Bạn muốn mình tư vấn thêm về mẫu nào không? 😊"
    else:
        return "Xin lỗi bạn nha, hiện tại shop chưa có sản phẩm phù hợp. Bạn thử tìm màu hoặc size khác xem, mình sẵn sàng tư vấn thêm! 😊"

# ==================== FLASK ROUTES ====================
@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    response = generate_response(user_input)
    print(f"User input: '{user_input}', Response: '{response}'")  # Debug log
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)