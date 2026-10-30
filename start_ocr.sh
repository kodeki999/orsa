#!/usr/bin/sh

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

. OE/bin/activate

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
