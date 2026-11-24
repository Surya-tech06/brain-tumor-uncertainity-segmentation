.PHONY: install test experiments app train-unet
install:        ; pip install -r requirements.txt
test:           ; python -m pytest -q
experiments:    ; python scripts/run_experiments.py
app:            ; python app/app.py
train-unet:     ; python scripts/train_unet.py --data synthetic --epochs 15
