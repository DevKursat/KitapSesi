#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Python ortamı bulunamadı."
  echo "Önce README.md içindeki kurulum adımlarını çalıştırın."
  read "?Kapatmak için Enter'a basın..."
  exit 1
fi

if ! .venv/bin/python -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "Arayüz bağımlılıkları eksik."
  echo "Çalıştırın: .venv/bin/python -m pip install -r requirements.txt"
  read "?Kapatmak için Enter'a basın..."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

(sleep 2; open "http://127.0.0.1:8000") &
exec caffeinate -dimsu .venv/bin/python -m uvicorn \
  web_app:app --host 127.0.0.1 --port 8000
