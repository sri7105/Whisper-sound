import json
import pandas as pd
import jiwer

def run_evaluation():
    df = pd.read_csv("results/predictions.csv")

    wer = jiwer.wer(df["ground_truth"].tolist(), df["prediction"].tolist())
    cer = jiwer.cer(df["ground_truth"].tolist(), df["prediction"].tolist())
    avg_latency = df["latency_sec"].mean()
    num_samples = len(df)

    metrics = {
        "num_samples": num_samples,
        "word_error_rate": round(wer, 4),
        "character_error_rate": round(cer, 4),
        "average_latency_sec": round(avg_latency, 4)
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("results/report.md", "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("Model: openai/whisper-small\n\n")
        f.write("Dataset: LibriSpeech ASR (clean, test split)\n\n")
        f.write(f"Samples processed: {num_samples}\n\n")
        f.write(f"Word Error Rate: {metrics['word_error_rate']}\n\n")
        f.write(f"Character Error Rate: {metrics['character_error_rate']}\n\n")
        f.write(f"Average latency (sec): {metrics['average_latency_sec']}\n")

    print(metrics)

if __name__ == "__main__":
    run_evaluation()