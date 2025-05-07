# ⚖️ Legal Document Summarizer

A powerful and intuitive web app built with **Streamlit** for summarizing legal documents such as Non-Disclosure Agreements (NDAs). It leverages **spaCy**, **pdfplumber**, and custom heuristics to extract key information, clean legal text, and generate concise summaries of uploaded PDF documents.

---

## 🚀 Features

- 📄 **PDF Upload Support** – Upload scanned or text-based legal documents in PDF format.
- ✂️ **Smart Summarization** – Extracts and ranks sentences using legal term frequency, sentence structure, and positioning.
- 🔍 **Metadata Extraction** – Automatically detects key metadata like:
  - Date of the agreement
  - Sender and receiver parties
  - Subject or purpose of the NDA
- 📚 **Clean Output** – Summarized content is cleaned, formatted, and presented as bullet points.
- ⬇️ **Download Option** – Easily download the summarized results as a `.txt` file.
- 🧠 **Built with NLP** – Uses `spaCy` and custom logic for accurate sentence scoring and preprocessing.

---


## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) – UI and app framework
- [spaCy](https://spacy.io/) – Natural Language Processing
- [pdfplumber](https://github.com/jsvine/pdfplumber) – PDF text extraction
- Python (3.8+)

---

## 🏁 Getting Started

### 🔧 Prerequisites

Make sure you have Python installed (>=3.8) and the following libraries:

```bash
pip install streamlit spacy pdfplumber
python -m spacy download en_core_web_sm
