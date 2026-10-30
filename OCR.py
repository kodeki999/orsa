from PIL import Image
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText, Pipeline,  AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import requests
device = Accelerator().device
# import os
import torch
import spacy
from pathlib import Path
import re

# Caricamento una sola volta quando parte il servizio
nlp = spacy.load("it_core_news_sm")

# MODEL_ID = "zai-org/GLM-OCR"
MODEL_ID = Path("models/GLM-OCR").resolve().as_posix()


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# Caricamento modello (fallo una sola volta all'avvio)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # esempio

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

prompt = """
Esegui l'OCR COMPLETO dell'immagine.

L'immagine contiene una busta paga. Devi trascrivere TUTTO il
contenuto visibile dell'immagine, dall'alto verso il basso e
da sinistra verso destra.

NON fare un riassunto.
NON selezionare solo le informazioni importanti.
NON omettere intestazioni, descrizioni, numeri, date, codici,
importi, totali o note.

È particolarmente importante preservare la relazione tra
le descrizioni e i relativi valori.

Quando il documento contiene una tabella:
- mantieni l'ordine delle colonne;
- mantieni l'ordine delle righe;
- mantieni ogni valore nella stessa riga della relativa descrizione;
- separa chiaramente le colonne;
- non spostare valori tra righe diverse.

Usa questo formato per le tabelle:

DESCRIZIONE | VALORE | ALTRA COLONNA | ...

Trascrivi anche il testo che non appartiene alle tabelle.

Non interpretare i dati.
Non correggere i dati.
Non effettuare calcoli.
Non inventare informazioni.

L'obiettivo è ottenere una trascrizione completa e fedele
dell'intera immagine, preservando il più possibile la
struttura visiva originale.
"""
def glm_ocr(image_path, num_token, is_table):

    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto"
    )
    image = Image.open(image_path).convert("RGB")
    if is_table:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Table Recognition:"},
                    {"type": "image", "image": image}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}
                ]
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=num_token,
            do_sample=False
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]



    return result




def SplitOllama(prompt, num_token=2500):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_token,
            "temperature": 0
        }
    }


    response = requests.post(
        OLLAMA_URL,
        json=payload
    )



    response.raise_for_status()

    result = response.json()

    return result["response"]



def verify_hug(prompt, num_token=2000):


    """
    Genera una risposta usando Hugging Face Transformers
    equivalente alla chiamata Ollama.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=num_token,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Estrae solo la parte generata, senza il prompt
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


def separa_maiuscole(testo):

    testo = re.sub(
        r"(?<=[a-zàèéìòù\.])\s+(?=[A-ZÀÈÉÌÒÙ][a-zàèéìòù]{3,}\b)",
        ". \n\n",
        testo
    )

    return testo


PATTERN = re.compile(
    r"(?<=\.)\s+(?=[A-ZÀÈÉÌÒÙ][a-zàèéìòù]{3,}\b)"
)


def split_nlp(testo: str) -> list:
    """
    Divide un testo in frasi usando spaCy.
    Corregge abbreviazioni tipiche HSE/OCR e
    riconosce anche l'inizio di nuove frasi senza punteggiatura.
    """

    if not testo:
        return []

    testo = re.sub(
        r"\s*_{5,}\s*",
        ".\n\n",
        testo
    )

    # Correzioni / separazioni specifiche HSE-OCR
    testo = testo.replace("non trovato", "\n")

    separatori = [
        "Si",
        "SI",
        "Come",
        "Come richiesto",
        "Ripristinare",
        "Si richiede",
        "Nell'esecuzione",
        "Delimitare",
        "Collocare",
        "Modificare",
    ]

    for parola in separatori:
        testo = testo.replace(
            f" {parola}",
            f" .\n\n {parola}"
        )

    # Protezione abbreviazioni
    sostituzioni = {
        "d.lgs.": "d_lgs",
        "s.m.i.": "s_m_i",
        "C.S.E.": "C_S_E",
        "P.S.C.": "P_S_C",
        "P.O.S.": "P_O_S",
    }

    testo_pulito = testo

    for vecchio, nuovo in sostituzioni.items():
        testo_pulito = testo_pulito.replace(vecchio, nuovo)

    # spaCy
    doc = nlp(testo_pulito)

    frasi = []

    for sent in doc.sents:
        frase = sent.text.strip()

        if not frase or frase == ".":
            continue

        # Ripristino abbreviazioni
        for nuovo, vecchio in {
            "d_lgs": "d.lgs.",
            "s_m_i": "s.m.i.",
            "C_S_E": "C.S.E.",
            "P_S_C": "P.S.C.",
            "P_O_S": "P.O.S.",
        }.items():
            frase = frase.replace(nuovo, vecchio)

        frasi.append(frase)

    # Secondo livello:
    # separa frasi attaccate senza punteggiatura
    frasi_finali = []

    for frase in frasi:
        parti = PATTERN.split(frase)

        for parte in parti:
            parte = parte.strip()

            if not parte or parte == ".":
                continue

            if parte:
                frasi_finali.append(parte)

    return frasi_finali