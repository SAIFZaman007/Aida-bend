"""Download the Piper voice models A.I.D.A uses, into backend/voices/.

Run once from the backend folder:
    python download_voices.py

Fetches a female (Amy) and a male (Ryan) US-English medium voice from the
official Piper voices repository on Hugging Face. Each voice is two files:
the model (.onnx) and its config (.onnx.json). ~60 MB each, CPU-only, free.

Want a different voice? Browse https://huggingface.co/rhasspy/piper-voices
and change the URLs / filenames below (also update PIPER_VOICE_* in .env).
"""
import os
import urllib.request

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"

VOICES = {
    "en_US-amy-medium": f"{BASE}/amy/medium/en_US-amy-medium.onnx",      # female
    "en_US-ryan-medium": f"{BASE}/ryan/medium/en_US-ryan-medium.onnx",   # male
}

OUT = os.path.join(os.path.dirname(__file__), "voices")


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  already have {os.path.basename(dest)}")
        return
    print(f"  downloading {os.path.basename(dest)} ...")

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r    {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _progress)
    print("\r    done   ")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, model_url in VOICES.items():
        print(name)
        _download(model_url, os.path.join(OUT, f"{name}.onnx"))
        _download(model_url + ".json", os.path.join(OUT, f"{name}.onnx.json"))
    print("\nVoices ready in:", OUT)
    print("Restart the backend, then set Settings -> Voice -> Engine to 'Piper'.")


if __name__ == "__main__":
    main()