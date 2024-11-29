import streamlit as st
from PIL import Image

from model_meme import detect_meme
from model_violence import detect_violence
from model_segment import segment_text
    

# --------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

image_preview = None
image_model = None



for uploaded_file in uploaded_files:  # pyright: ignore[]
    st.write("filename:", uploaded_file.name)
    # Check if the file is an image
    if uploaded_file.name.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp")):
        image_preview = Image.open(uploaded_file)
        image_model = uploaded_file  # use for model


# Setup layout
image_preview_area, generate_result_area = st.columns(2)

# Check if an image is available for preview
if image_preview and image_model:
    # Display image in the first column
    with image_preview_area:
        st.image(image_preview, caption="Uploaded Image", use_container_width=True)  # pyright: ignore[]

    # Create two columns
    end_loop = True
    while end_loop:
        with st.spinner("Wait for it..."):
            with generate_result_area:
                # get text
                text_segment = segment_text(image_model=image_model)
                result_detect_meme = detect_meme(
                    image_model=image_model, text=text_segment
                )
                print("result_detect_meme", result_detect_meme)
                if result_detect_meme["detect_meme"][0][0].item() == 1:
                    st.text("Is violence/harmfull image")
                    break
                else:
                    st.text("None violence/harmfull image ")
                    break
                    result_detect_violence = detect_violence(image_model=image_model)  # pyright: ignore[]
                    print("result_detect_violence", result_detect_violence)
                    if (
                        result_detect_violence["violence_pred"] == 1
                        and result_detect_violence["human_pred"] == 1
                    ):
                        st.text("Is violence/harmfull image")
                        break
                    else:
                        st.text("Is non violence/harmfull image")
                        break

else:
    st.write("No image file uploaded.")
