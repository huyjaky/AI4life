from transformers import AutoModel, AutoTokenizer
import pytesseract
from PIL import Image

def segment_text(image_model):
    image = Image.open(image_model).convert("RGB")
    return pytesseract.image_to_string(image)

