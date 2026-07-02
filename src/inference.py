import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def run_inference():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()

    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = list(dataset.take(30))

    results = []

    for sample in samples:
        audio = sample["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        latency = time.time() - start

        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        results.append({
            "audio_id": sample["id"],
            "ground_truth": sample["text"],
            "prediction": prediction.strip().upper(),
            "latency_sec": latency
        })

    df = pd.DataFrame(results)
    df.to_csv("results/predictions.csv", index=False)

if __name__ == "__main__":
    run_inference()