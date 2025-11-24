# Transcript Scorer

Transcript Scorer is an AI‑powered application that evaluates transcripts (interview transcripts, call summaries, meeting recordings, etc.) against a scoring rubric. You can either use the **default rubric** provided in the application or upload **your own custom rubric** for scoring.

The system supports transcript inputs in **PDF**, **TXT**, or plain text formats and produces structured scoring results along with feedback.

---

## ✨ Features

### **1. Score Transcripts Using Default Rubric**

A preloaded rubric (stored in `data/default_rubric.xlsx`) is used when no custom rubric is provided.

### **2. Custom Rubric Scoring**

Users can upload their own rubric (Excel-based or text-based), and the AI judges the transcript based on that rubric's criteria.

### **3. Clean, Format, and Validate Rubrics**

The **Rubric Formatter Agent** ensures that uploaded rubrics are:

* Structured
* Normalized
* Ready for consistent evaluation

### **4. Intelligent Transcript Parsing**

The app automatically extracts text from PDF or TXT files using the `file_parser` utility.

### **5. AI‑Driven Scoring**

The **Scoring Agent** evaluates the transcript using LLM reasoning to generate:

* Criterion-level scores
* Weighted totals
* Detailed feedback

### **6. Orchestration Layer**

The **Orchestrator Agent** coordinates the entire pipeline:

* Loading rubric
* Cleaning rubric
* Parsing transcript
* Running scoring agent
* Merging and formatting results

### **7. UI + API Support**

* **Streamlit UI** for interactive use
* **FastAPI backend** for programmatic scoring via API calls

---

## 🗂 Project Structure

```
transcript-scorer/
├── .env                      # API keys
├── requirements.txt
├── app.py                    # Streamlit UI
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints
│   └── models.py            # Pydantic schemas
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Main orchestrator
│   ├── rubric_formatter.py  # Rubric cleaning agent
│   └── scorer.py            # Scoring agent
├── utils/
│   ├── __init__.py
│   ├── file_parser.py       # PDF/TXT extraction
│   └── rubric_loader.py     # Excel/uploaded rubric handler
└── data/
    └── default_rubric.xlsx   # Default scoring rubric
```

---

## 🚀 Setup Instructions

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/transcript-scorer.git
cd transcript-scorer
```

### **2. Install Dependencies**

```bash
uv sync
```

### **3. Add API Keys**

Add your Gemini LLM keys to the `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

### **4. Run the Streamlit App**

```bash
streamlit run app.py
```

### **5. Run the FastAPI Server**

```bash
uvicorn api.main:app --reload
```

---

## 🔧 API Usage Example

**POST /score**

```json
{
  "transcript": "Your transcript text...",
  "rubric": "Optional custom rubric text"
}
```

---

## 📌 TODO / Roadmap

* [ ] Add PDF report export
* [ ] Add score visualization charts
* [ ] Add batch scoring support

---

## 📄 License

MIT License.

---
