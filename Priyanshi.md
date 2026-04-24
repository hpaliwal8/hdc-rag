!git clone https://github.com/hpaliwal8/hdc-rag.git
%cd hdc-rag

!pip install -U bitsandbytes>=0.46.1

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/hdc-rag-outputs"

from huggingface_hub import login
login()


(before running this, Go to this page while logged into your HuggingFace account:

huggingface.co/meta-llama/Llama-3.1-8B-Instruct

You'll see a license agreement form — fill in your details and click Agree. Meta usually approves access within a few minutes to a few hours.\

That's it.)

!python scripts/run_experiments.py --config config/default.yaml --dataset hotpotqa --limit 500 --output "/content/drive/MyDrive/hdc-rag-outputs/hotpotqa_llama31.jsonl"
