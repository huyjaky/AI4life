
import streamlit as st
from PIL import Image

from model_meme import detect_meme
from model_violence import detect_violence
from model_segment import segment_text

st.set_page_config(layout="wide")

list_result = []

# NOTE: result { image_preview, result, image_name}
def show_result(result):
    list_result.append(result)

    image_preview_area, generate_result_area = st.columns(2)
    with image_preview_area:
        st.image(result['image_preview'], caption="Uploaded Image", use_container_width=True)  # pyright: ignore[]
    with generate_result_area:
        if result['result'] == 1:
            st.title("Harmful image")
        else: 
            st.title("Not harmful image")

def return_result(image_preview, image_name):
    image = Image.open(image_preview)
    text_segment = segment_text(image_model=image_preview)
    result_detect_meme = detect_meme(
        image_model=image_preview, text=text_segment
    )
    print("result_detect_meme", result_detect_meme)
    return {"image_preview": image, "result": result_detect_meme["detect_meme"][0][0].item(), "image_name": image_name}



uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

image_preview = None


for uploaded_file in uploaded_files:  # pyright: ignore[]
    st.write("filename:", uploaded_file.name)

    # Check if the file is predicted
    for result in list_result:
        image_name = result['image_preview']
        if image_name == uploaded_file.name:
            continue
    if uploaded_file.name.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp")):
        result = return_result(uploaded_file, uploaded_file.name)
        show_result(result)

