# Pashto TTS Model Training Experiments

This repository collects a series of Jupyter notebooks and training scripts where we attempted to build Pashto Text-to-Speech systems via different model architectures and pipelines. Each notebook represents one experiment: data preprocessing, model setup, training, and inference. Below is a chronological walkthrough of what we tried, why each approach didn’t meet quality targets, and key lessons learned.

---

## Table of Contents

1. [try_1.ipynb](#1-try_1ipynb)  
2. [tts_training_pashto.ipynb](#2-tts_training_pashtoipynb)  
3. [Chinese (1) (1).ipynb](#3-chinese-1-1ipynb)  
4. [Chinese (1).ipynb](#4-chinese-1ipynb)  
5. [lasttry.ipynb](#5-lasttryipynb)  
6. [helloworld.ipynb](#6-helloworldipynb)  
7. [KP.ipynb](#7-kpipynb)  
8. [lol.ipynb](#8-lolipynb)  
9. [mariah.ipynb](#9-mariahipynb) *(Ongoing)*  

---

### 1. `what.ipynb`

- **Model**  
  - Hugging Face **Indic-Parler-TTS** checkpoint  
    ```python
    ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts")
    ```
- **Pipeline Highlights**  
  - Custom Pashto tokenizer (char / BPE / word)  
  - Audio download & preprocessing (resample, normalize)  
  - `Seq2SeqTrainer` for text→audio-code prediction  
  - Decoding quantized codes via the model’s DAC decoder → WAV  
- **Key Failure Points**  
  1. **Tokenizer mismatch**: Pretraining on Indian languages → poor Pashto phoneme coverage  
  2. **Data volume**: Fine-tuning on limited hours → overfitting rare tokens  
  3. **No dedicated vocoder**: Built-in decoder fails on unseen phonetics  

---

### 2. `tts_training_pashto.ipynb`

- **Model**  
  - Microsoft **SpeechT5** TTS  
    ```python
    SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    ```
- **Pipeline Highlights**  
  - `SpeechT5Processor` for feature extraction & tokenization  
  - Full-stack fine-tuning of encoder + decoder on Pashto text/audio  
- **Key Failure Points**  
  1. **Over-adaptation**: Full fine-tuning washes out pretrained speech priors on scarce data  
  2. **Default hyperparameters**: Learning rate and epochs too aggressive for low-resource language  
  3. **Vocoder gap**: No separate Hi-Fi GAN stage → poor mel→waveform quality  

---

### 3. `Chinese (1) (1).ipynb`

- **Model**  
  - SpeechT5 backbone, loaded from local Pashto checkpoint  
    ```python
    SpeechT5ForTextToSpeech.from_pretrained("./speecht5_tts_pashto_final")
    ```
- **Pipeline Highlights**  
  - Phrase-level tokenization via 🤗 datasets  
  - Audio cleaning with `noisereduce`  
  - Mixed imports: SpeechT5 Hi-Fi GAN, Coqui TTS API, VITS training stub  
- **Key Failure Points**  
  1. **Overly complex pipeline**: Multiple frameworks & vocoders → feature misalignments  
  2. **Tokenizer → positional-embedding mismatch** when changing token boundaries  
  3. **Cleaning artifacts**: Noise reduction distorts spectral cues expected by vocoder  

---

### 4. `Chinese (1).ipynb`

- **Model**  
  - Microsoft SpeechT5 + Hi-Fi GAN vocoder  
    ```python
    SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    ```
- **Pipeline Highlights**  
  - Pashto fine-tuning of SpeechT5  
  - Hi-Fi GAN for mel→waveform synthesis  
- **Key Failure Points**  
  1. **Spectrogram scaling**: Mismatch between model output amplitudes & vocoder input  
  2. **Data cleaning**: Aggressive noise suppression → muffled audio  
  3. **Layer-freezing**: No controlled freezing strategy → unstable fine-tuning  

---

### 5. `lasttry.ipynb`

- **Model**  
  - Fully **custom** `PashtoTTSModel` (no pretraining)  
    - `TTSEncoder` → `TTSDecoder`
- **Pipeline Highlights**  
  - `PashtoTTSDataset`, `AdvancedTTSTrainer` for end-to-end training  
- **Key Failure Points**  
  1. **No pretraining**: Custom from-scratch underfits on limited Pashto data  
  2. **Under-parameterized**: Encoder/decoder depth insufficient for natural prosody  
  3. **Training regime**: Single LR, no guided-attention or duration predictor → poor alignment  

---

### 6. `helloworld.ipynb`

- **Model**  
  - Microsoft SpeechT5 with **progressive fine-tuning** & **hybrid decoder**  
- **Pipeline Highlights**  
  - “Progressive 10K” incremental training rounds  
  - Hybrid: swap in Pashto-trained encoder, freeze portions of decoder  
  - Custom collators (`FixedTTSDataCollator`, `OptimizedTTSDataCollator`)  
- **Key Failure Points**  
  1. **Over-freezing**: Preserved audio quality but prevented Pashto prosody adaptation  
  2. **Curriculum schedule**: No dynamic LR or data-sampling → catastrophic forgetting  
  3. **Collator mismatches**: Padding/normalization misaligned mel scales  

---

### 7. `KP.ipynb`

- **Model**  
  - Default Coqui TTS API demo (English Tacotron2-DDC)  
- **Pipeline Highlights**  
  - Smoke test: `from TTS.api import TTS`  
- **Key Failure Points**  
  1. **No Pashto checkpoint**: Used English LJSpeech model by default  
  2. **No training/synthesis**: Pipeline never executed beyond import  

---

### 8. `lol.ipynb`

- **Model**  
  - Fully **custom** `SimpleTTSModel` (LSTM encoder + attention + LSTM decoder)  
- **Pipeline Highlights**  
  - Autoregressive mel spectrogram prediction  
  - Griffin-Lim–style vocoder in `robust_mel_to_audio()`  
- **Key Failure Points**  
  1. **No pretrained backbone** → severe under-learning on limited data  
  2. **Unstable attention**: No guided-attention or duration model → alignment failures  
  3. **Basic vocoder**: Griffin-Lim yields muffled, artifact-ridden audio  

---

### 9. `mariah.ipynb`  *(Ongoing)*

- **Model**  
  - Coqui TTS via external scripts (`train_pashto_tts.py`, `_fixed`, `_safe`)  
- **Pipeline Highlights**  
  - Defines Windows batch & Python entry points  
- **Key Failure Points**  
  1. **No in-notebook feedback**: No loss curves or sample audio for early debugging  
  2. **Ambiguous configs**: Multiple script variants without a clear standard  
  3. **OS-hardcoding**: Windows paths hinder Linux/CI reproducibility  

---

## Next Steps & Recommendations

- **Pick a single backbone** (e.g., SpeechT5) and establish a minimal, reproducible fine-tuning recipe.  
- **Align data pipeline**: Verify mel spectrogram scales and feature-normalization match vocoder specs.  
- **Controlled freezing**: Start by freezing encoder layers, then gradually unfreeze decoder blocks.  
- **Add duration predictor or guided-attention**: Consider FastSpeech2 or Tacotron2 for robust alignment.  
- **In-notebook monitoring**: Log loss curves and sample audio at checkpoints for rapid iteration.  

---

**Credits & Licensing**  
- Uses open-source TTS libraries: Hugging Face Transformers, Coqui TTS, PyTorch.  
- Please cite respective model papers and repositories when reusing these notebooks.  
