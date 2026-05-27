# Meta-prompt for Codex (GPT)

## Purpose
This meta-prompt is given to Codex (GPT) to generate two task prompts (English and Russian) for a botnet detection code generation experiment. The generated prompts will then be tested on three LLMs: Claude, DeepSeek, and GigaChat.

## Meta-prompt

---

You are an expert in prompt engineering for cybersecurity and machine learning tasks. Your goal is to generate two prompts — one in English and one in Russian — that will be given to different LLMs to generate a complete Python script for botnet detection.

The prompts must follow the 4-component methodology from Monge Martinez (2023) "Using LLMs and GPT to streamline data analysis in cybersecurity incidents":

1. **Role** — define who the LLM is (a Python code generator for cybersecurity)
2. **Context** — explain the problem (anomaly-based botnet detection using autoencoders on IoT traffic)
3. **Data** — describe the dataset structure so the LLM knows what it's working with
4. **Steps** — provide a concrete sequence of actions the generated script must perform

Use the following information to compose both prompts:

**Dataset**: N-BaIoT (UCI ML Repository, ID 442)
- Real network traffic from 9 IoT devices (cameras, doorbells, thermostats)
- Infected with two botnets: Mirai and BASHLITE (Gafgyt)
- 115 numerical features per record (network flow statistics)
- CSV files organized as:
  - `<device_id>.benign.csv` — normal traffic
  - `<device_id>.gafgyt.<attack>.csv` — BASHLITE attacks (combo, junk, scan, tcp, udp)
  - `<device_id>.mirai.<attack>.csv` — Mirai attacks (ack, scan, syn, udp, udpplain)
- Reference paper: Meidan et al. (2018) "N-BaIoT — Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders"

**Approach**: Train a deep autoencoder only on benign traffic. Use reconstruction error (MSE) as anomaly score. Traffic with MSE above threshold = attack.

**Required script steps**:
1. Load benign + attack CSVs for a configurable device, label them (0=benign, 1=attack)
2. Split: 2/3 benign for training, 1/3 benign + all attacks for testing; 80/20 train/validation split; MinMaxScaler fitted on train only; handle NaN/Inf
3. Build deep autoencoder (Keras): encoder 4 layers (100%-75%-50%-33% of 115), each Dense+BatchNorm+LeakyReLU; bottleneck 25%; symmetric decoder; linear output; MSE loss, Adam optimizer; EarlyStopping + ReduceLROnPlateau
4. Train autoencoder (input=target, benign only)
5. Threshold = mean(val_MSE) + std(val_MSE)
6. Evaluate: classification report, accuracy, confusion matrix
7. Visualize: loss curves + reconstruction error histogram (benign vs attack)
8. Print total execution time

**Requirements for the generated prompts**:
- Both prompts must be semantically identical (same task, same requirements)
- The English prompt should use natural technical English
- The Russian prompt should use natural technical Russian (not a mechanical translation)
- Both must be self-contained — an LLM reading only the prompt should produce a working script
- Use argparse for device ID and data folder path
- Request well-commented code with progress output

Generate both prompts now. Label them clearly: "ENGLISH PROMPT" and "RUSSIAN PROMPT" (ПРОМПТ НА РУССКОМ ЯЗЫКЕ).

---
