from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np


# DEFAULT_DIMENSION = 64
# DEFAULT_BATCH_SIZE = 64
# DEFAULT_MAX_LENGTH = 512
MODEL_PATH = "models/bge-m3"



dim = 1024
torch.set_float32_matmul_precision("high")

model = BGEM3FlagModel(
    MODEL_PATH,
    use_fp16=True,
    devices="cuda",
)

modelDense = SentenceTransformer(
    MODEL_PATH,
    device="cuda"
)

def modDim(intern):
    global dim
    dim = intern
 
def getDim():
    return dim

def embDense(chunks):

    embeddings = modelDense.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        truncate_dim=dim 
    )

    result = {}

    for parola, vec in zip(chunks, embeddings):
        result[parola] = vec.tolist()

    return result



def embSparse(chunks):

    model = BGEM3FlagModel(
        "/orsa/models/bge-m3",
        use_fp16=True,
        devices="cuda"
        )


    embeddings = model.encode(
        chunks,
        batch_size=64,
        max_length=512,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False
    )

    sparse_embeddings = embeddings["lexical_weights"]

    result = {}

    for testo, sparse_vector in zip(chunks, sparse_embeddings):
        result[testo] = {
            "indices": [
                int(token_id)
                for token_id in sparse_vector.keys()
            ],
            "values": [
                float(weight)
                for weight in sparse_vector.values()
            ]
        }

    return result


def cosSim(target, opt):
    
    model = SentenceTransformer(
        "/orsa/models/bge-m3",
        device="cuda"
    )

    buffer = [target] + opt 

    emb = model.encode(
        buffer,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    target_vec = emb[0]
    other_vecs = emb[1:]


    result = {}
    for parola, vec in zip(opt, other_vecs):
        cos_sim = float(np.dot(target_vec, vec))
        angolo = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        result.update({parola : angolo})
        

    return result

