from transformers import AutoModel, AutoTokenizer
import pytesseract
from PIL import Image

# def segment_text(image_model):
#     image = Image.open(image_model).convert("RGB")
#     return pytesseract.image_to_string(image)

# print(type(segment_text('./p072ms6r.jpg')))
# print(segment_text('./p072ms6r.jpg'))

def segment_text(image_model):
    image = Image.open(image_model).convert("RGB")
    
    # Chạy OCR để nhận diện văn bản từ ảnh
    text = pytesseract.image_to_string(image)
    
    # Kiểm tra nếu văn bản có thể được mã hóa UTF-8
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')  # Chuyển đổi chuỗi văn bản
    except UnicodeDecodeError:
        print("There was an error decoding the text.")

    return {"segment_text": text}

