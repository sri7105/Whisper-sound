# Whisper-sound
# Whisper ASR Pipeline

## Overview
This project runs inference using openai/whisper-small on 30 samples from the
LibriSpeech ASR (clean, test split) dataset, then evaluates predictions using
Word Error Rate (WER), Character Error Rate (CER), and average inference latency.

## Model
openai/whisper-small (encoder-decoder Transformer, autoregressive decoding)

## Dataset
LibriSpeech ASR, "clean" config, test split, loaded via streaming mode
(only downloads the samples used, not the full archive)

## Setup
pip install -r requirements.txt

## Run
python run.py

This will:
1. Download the Whisper-small model and processor from Hugging Face
2. Stream 30 samples from LibriSpeech ASR (clean/test)
3. Run inference and save results/predictions.csv
4. Compute evaluation metrics and save results/metrics.json and results/report.md

## Results
- Samples processed: 30
- Word Error Rate: 0.1639
- Character Error Rate: 0.0365
- Average inference latency: 4.51 sec/sample (CPU)

Note: WER is higher than CER here because ground truth text in LibriSpeech is
uppercase with no punctuation, while Whisper's raw output includes punctuation.
This inflates word-level error count even when the words themselves are correct.
Character-level comparison (CER) is less affected by this and better reflects
actual transcription accuracy.

## Implementation notes
- datasets is pinned to 2.19.0 to avoid a torchcodec/FFmpeg dependency issue
  with audio decoding on Windows; this version decodes audio via librosa/soundfile.
- Streaming mode is used for the dataset to avoid downloading the full
  LibriSpeech archive (several GB) when only 30 samples are needed.

## Project structure
whisper-asr-pipeline/
├── README.md
├── requirements.txt
├── run.py
├── research_summary.docx
├── src/
│   ├── inference.py
│   └── evaluate.py
└── results/
    ├── predictions.csv
    ├── metrics.json
    └── report.md

Initial attempts to load the dataset without streaming triggered a download of the full LibriSpeech archive (several GB), which is unnecessary when only a small sample subset is needed, switching to streaming mode fixed this.
