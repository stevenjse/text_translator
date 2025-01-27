import gradio as gr
import easyocr
import torch
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
from transformers import AutoModelForSequenceClassification
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Cargar el modelo y tokenizer de InternVL
path = 'OpenGVLab/InternVL2_5-2B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Cargar el modelo de detección de idioma
model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer_ld = AutoTokenizer.from_pretrained(model_ckpt)
model_ld = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

# Función para realizar OCR con EasyOCR
def extract_text_easyocr(image):
    # reader = easyocr.Reader(['en', 'es', 'fr'], gpu=True)
    # Inicializar EasyOCR para solo idiomas latinos
    reader_latin = easyocr.Reader(['fr', 'de', 'en', 'es', 'it'], gpu=True)
    results = reader_latin.readtext(image)
    extracted_text = ""
    for _, text, _ in results:
        extracted_text += text + " "
    return extracted_text

# Función para realizar OCR con InternVL
def extract_text_internvl(image_path):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=512, do_sample=True)
    question = '<image>\n Give me the complete plain text in the image in the original language. Do not provide explanations, translations, or any additional text. Only the plain text.'
    # question = '<image>\n Give me the complete plain text in the image in the original language.'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    response = response.replace("\n", " ").strip()
    return response

# Función para combinar ambos resultados de OCR
def combine_ocr_responses(response_easyocr, response_internvl):
    #question = f'Combine {response_easyocr} and {response_internvl}, preserving their original alphabets and correcting any errors.'
    # question = f'Combine "{response_easyocr}" and "{response_internvl}" into a single coherent phrase, giving more importance to "{response_internvl}" as it comes from an LLM capable of detecting text within images. Ensure the original languages and alphabets of both inputs are preserved. Return only the combined phrase inside square brackets, like this: [combined text]. Do not translate, explain, or add any other text—only the phrase inside square brackets.'
    question = f'Combine "{response_easyocr}" and "{response_internvl}" into a single meaningful phrase, ensuring that both inputs are equally considered and their original alphabets are preserved. Do not include explanations, translations, or any additional text—only the corrected phrase.'
    # question = f'Combine {response_easyocr} and {response_internvl}, preserving their original alphabets and correcting any errors. Return only the corrected phrase inside square brackets, like this: [corrected text]. Do not provide any explanations, translations or any additional text.'
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    combined_response = model.chat(tokenizer, None, question, generation_config, history=None, return_history=False)
    return combined_response.replace("\n", " ").strip()

# Función para identificar el idioma del texto
def detect_language(text):
    inputs = tokenizer_ld(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_ld(**inputs).logits
    preds = torch.softmax(logits, dim=-1)
    id2lang = model_ld.config.id2label
    vals, idxs = torch.max(preds, dim=1)
    detected = {id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}
    return list(detected.keys())[0]

# Función para la traducción
def translate_text(text, original_language, target_language):
    model_name = f"Helsinki-NLP/opus-mt-{original_language}-{target_language}"
    tokenizer_t = MarianTokenizer.from_pretrained(model_name)
    model_t = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer_t(text, return_tensors="pt", padding=True)
    translated = model_t.generate(**inputs)
    tgt_text = [tokenizer_t.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text[0]

# Función para traducción con InternVL
def translate_internvl(text, original_language, target_language):
    question = f'The following text "{text}" it is in {original_language} language, translate it to {target_language} language. Remove any parts that are incoherent or do not make sense in the text. Do not include explanations or any additional text-only the translated text.'
    #question = f'The following text "{text}" it is in {original_language}, please translate it to {target_language}. Remove any parts that are incoherent or do not make sense in the text. Provide only the cleaned and translated text, without explanations, context, or additional comments.'
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    #pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, None, question, generation_config, history=None, return_history=False)
    return response.strip()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Función para cargar la imagen y procesarla (similar a tu código previo)
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Función para la interfaz de Gradio
def process_image(image, target_language="es"):
    if not image:
        return "Por favor, sube una imagen.", "", "", "", "", ""
    if not target_language:
        return "Por favor, selecciona un idioma objetivo.", "", "", "", "", ""

    # Extraer texto con EasyOCR
    text_easyocr = extract_text_easyocr(image)
    
    # Extraer texto con InternVL
    text_internvl = extract_text_internvl(image)
    
    # Combinar los resultados de EasyOCR e InternVL
    combined_text = combine_ocr_responses(text_easyocr, text_internvl)
    
    # Identificar el idioma del texto combinado
    detected_language = detect_language(combined_text)

    # Diccionario de idiomas
    languages = {
        "es": "spanish",
        "en": "english",
        "fr": "french",
        "de": "german",
        "it": "italian"
    }

    # Realizar la traducción
    translation_model = translate_text(combined_text, detected_language, target_language)
    translation_internvl = translate_internvl(combined_text, languages[detected_language], languages[target_language])
    
    return (text_easyocr, text_internvl, combined_text, detected_language, translation_model, translation_internvl)

# Crear la interfaz de Gradio
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Sube una imagen"),
        gr.Dropdown(choices=["es", "en", "fr", "de"], label="Selecciona el idioma objetivo"),
        #gr.Button("Procesar")  # Botón "Procesar" para iniciar el proceso
    ],
    outputs=[
        gr.Textbox(label="Texto extraído con EasyOCR"),
        gr.Textbox(label="Texto extraído con InternVL"),
        gr.Textbox(label="Texto combinado entre EasyOCR e InternVL"),
        gr.Textbox(label="Idioma identificado"),
        gr.Textbox(label="Traducción con modelo"),
        gr.Textbox(label="Traducción con InternVL")
    ],
    live=True,  # Desactivamos el modo "live"
    description="Carga una imagen para realizar OCR y traducción con EasyOCR e InternVL.",
    title="Extractor y Traductor de Texto"
)

iface.launch()