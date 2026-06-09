---
title: LE Domain Console
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# LE Domain Console

Streamlit app for legal-entity ↔ domain mapping and 3-check accuracy verification.

## Required Space Secrets

Set these under **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key |
| `GOOGLE_SA_JSON` | Full JSON contents of the Google service-account key (paste as one line) |
| `CH_API_KEYS` | Comma-separated Companies House API keys (optional, UK only) |

## Local run

```bash
pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```
