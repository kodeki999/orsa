#!/usr/bin/env python3

import OCR
import EMB
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import shutil
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd
import io
import ast
import re
import json


# inferenza = OCR.glm_ocr("test.png")

# print("c'è scritto: ", inferenza)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/healt")
async def healt():
    
    return {
        "result": "okay",
        "timestamp": "ora"
    }

class modDimRequest(BaseModel):
    dim: int

class splitRequest(BaseModel):
    text: str


class samples(BaseModel):
    original: str
    secondary: str


class cosRequest(BaseModel):
    target: str
    chunks: list[str]


class EmbRequest(BaseModel):
    chunks: list[str]

class SplitResponse(BaseModel):
    sentences: list[str]


@app.get("/embDim")
async def embDim():
    try:
        return {
            "vector_dim": EMB.getDim()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )



@app.post("/embModDim")
async def embDim(req: modDimRequest):
    try:
        EMB.modDim(req.dim)
        return {
            "vector_dim": EMB.getDim()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )



@app.post("/embSparse")
async def emb(req: EmbRequest):
    try:
        result = EMB.embSparse(req.chunks)
        return{
            "result": result,
            "timestamp" : datetime.now().isoformat(timespec="seconds")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )


@app.post("/embDense")
async def emb(req: EmbRequest):
    try:
        result = EMB.embDense(req.chunks)
        return{
            "result": result,
            "timestamp" : datetime.now().isoformat(timespec="seconds")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )


@app.post("/cosSim")
async def run_embeddings(req: cosRequest):
    try:
        result = EMB.cosSim(req.target, req.chunks)

        return {
            "result": result,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )

@app.post("/capcha")
async def run_ocr(
    image: UploadFile = File(...),
    filename: str = Form("input_image.png")
):
    try:
        # working directory corrente
        wd = Path.cwd()

        # nome file sicuro
        safe_name = Path(filename).name
        path_img = wd / safe_name

        # salva immagine nella working directory
        with path_img.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # lancia OCR
        result = OCR.glm_ocr(str(path_img), 2048, False)

        # timestamp risposta
        timestamp = datetime.now().isoformat(timespec="seconds")

        return {
            "result": result,
            "timestamp": timestamp
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )


@app.post("/table")
async def run_ocr(
    image: UploadFile = File(...),
    filename: str = Form("input_image.png")
):
    try:
        # working directory corrente
        wd = Path.cwd()

        # nome file sicuro
        safe_name = Path(filename).name
        path_img = wd / safe_name

        # salva immagine nella working directory
        with path_img.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # lancia OCR
        result = OCR.glm_ocr(str(path_img), 8192, True)

        # timestamp risposta
        timestamp = datetime.now().isoformat(timespec="seconds")

        return {
            "result": result,
            "timestamp": timestamp
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )



@app.post("/verify")
async def verify(
    controllo : UploadFile = File(...),
    versione : UploadFile = File(...)
):
    try:
    
        controllo_text = (await controllo.read()).decode("utf-8")
    
        versione_text = (await versione.read()).decode("utf-8")

        prompt = f"""
        Devi confrontare due file json.

        FILE CONTROLLO:
        ----------------
        {controllo_text}

        FILE VERSIONE:
        ----------------
        {versione_text}

        Restituisci solo le differenze trovate con una formattazione json.
        Ecco un esempio di formattazione json per le differenze:
        '''
        {{
            "differenze": [
                {{
                    "chiave": "nome_chiave",
                    "valore_controllo": "valore_file_controllo",
                    "valore_versione": "valore_file_versione"
                }},
                ...
            ]
        }}
        '''
        non scrivere altro testo, solo il json con le differenze.
        """


        result = OCR.verify(prompt, 5000)

        return {
            "result": result,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )

def pulisci_output_lista(output):
    """
    Pulisce un output testuale contenente una lista Python/JSON
    e restituisce una lista Python.
    """

    if not isinstance(output, str):
        raise TypeError("L'input deve essere una stringa")

    # Rimuove markdown ```python ... ```
    output = re.sub(
        r"```(?:python)?",
        "",
        output,
        flags=re.IGNORECASE
    )

    output = output.replace("```", "").strip()

    # Rimuove eventuali escape
    output = output.replace("\\n", "")
    output = output.replace('\\"', '"')

    # Cerca il blocco lista
    match = re.search(
        r"\[.*\]",
        output,
        flags=re.DOTALL
    )

    if not match:
        raise ValueError(
            "Nessuna lista trovata nell'output"
        )

    lista_testo = match.group(0)

    try:
        risultato = ast.literal_eval(lista_testo)

    except Exception as e:
        raise ValueError(
            f"Errore nel parsing della lista: {e}\n"
            f"Output:\n{lista_testo}"
        )

    if not isinstance(risultato, list):
        raise TypeError(
            "Il risultato non è una lista Python"
        )

    # Pulizia elementi
    risultato = [
        str(item).strip()
        for item in risultato
        if str(item).strip()
    ]

    return risultato

SPLIT_PROMPT = """
Sei un sistema specializzato nella segmentazione di testo.

Il tuo compito è dividere il testo fornito in singole sentenze
(frasi) mantenendo esattamente il significato e il contenuto
originale.

REGOLE:

1. Dividi il testo in singole sentenze.
2. Ogni elemento della lista deve contenere una sola sentenza.
3. Mantieni il testo originale il più possibile.
4. Non riassumere.
5. Non correggere la grammatica.
6. Non modificare le parole.
7. Non aggiungere informazioni.
8. Non eliminare informazioni.
9. Mantieni la punteggiatura originale quando possibile.
10. Mantieni le maiuscole e minuscole originali.
11. le frasi tra parentesi vanno considerate come parte della frase principale.
12. Non dividere una frase in più parti se non costituisce
   effettivamente una sentenza indipendente.
13. Non inserire spiegazioni.
14. Non usare Markdown.
15. Restituisci esclusivamente una lista Python valida.
16. Ogni elemento della lista deve essere una stringa racchiusa
    tra virgolette doppie.


Esempio:

TESTO:
"Marco è andato al supermercato. Ha comprato il pane e il latte.
Poi è tornato a casa."

OUTPUT:
["Marco è andato al supermercato.", "Ha comprato il pane e il latte.", "Poi è tornato a casa."]

Ora segmenta il seguente testo:

TESTO:
{text}

OUTPUT:
"""


def sistema_stringa(testo):
    testo = testo.strip()

    if not testo.startswith('["'):
        testo = '["' + testo

    if not testo.endswith('"]'):
        testo = testo + '"]'

    return testo


@app.post("/split")
async def split(request: splitRequest):
    try:
        print(request.text)
        prompt = SPLIT_PROMPT.format(
            text=request.text
        )
        result = OCR.SplitOllama(prompt)
        print("===== RISPOSTA MODELLO =====")
        print(result)
        print("============================")
        result = sistema_stringa(result)
        result = result.replace('"non trovato",', '')
        lista = json.loads(result)

        print("===== RISULTATO PULITO =====")
        print(lista)
        return {
            "result": lista,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error":  traceback.format_exc(),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        )


@app.post("/splitGen", response_model=SplitResponse)
def split_gen(request: splitRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Il testo da segmentare non può essere vuoto."
        )

    prompt = SPLIT_PROMPT.format(
        text=request.text
    )

    try:
        # Generazione del modello
        output = verify_hug(
            prompt,
            num_token=2000
        )

        # Parsing e pulizia
        sentences = pulisci_output_lista(output)

        return SplitResponse(
            sentences=sentences
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante la segmentazione: {str(e)}"
        )
