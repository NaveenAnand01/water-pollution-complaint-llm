# 💧 LLM-Based Automated Water Pollution Complaint Classification and Routing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaveenAnand01/water-pollution-complaint-llm/blob/main/LLM_Water_Pollution_Complaint_Routing.ipynb)

An end-to-end system that reads citizens' free-text water pollution complaints, **classifies** them with a Large Language Model, assigns a **priority and SLA**, and **routes** each one to the responsible government department. An interactive **Gradio + Plotly dashboard** runs inside Google Colab.

## Pipeline

```
Citizen complaint (free text)
        │
        ▼
┌──────────────────────┐   few-shot prompt, 8 categories
│  LLM classifier      │── Qwen2.5-Instruct (open-source, free on Colab T4)
│  (answer-token probs)│   → category + confidence
└──────────┬───────────┘
           ▼
┌──────────────────────┐   TF-IDF + Logistic Regression
│  Priority model      │── + safety-keyword escalation
└──────────┬───────────┘   → High / Medium / Low + SLA (24 / 72 / 168 h)
           ▼
┌──────────────────────┐   category → department
│  Routing engine      │── confidence < 0.55 → human review
└──────────┬───────────┘
           ▼
     📊 Dashboard (live classifier · overview · routing queue · model performance)
```

## Dataset: **AquaGrievance-TN, a water pollution complaint dataset**

| | |
|---|---|
| File | [`data/water_pollution_complaints.csv`](data/water_pollution_complaints.csv) |
| Rows | 2,500 complaints (1,753 train / 747 test) |
| Coverage | 12 Tamil Nadu districts, 40+ rivers, lakes and canals, Jan 2025 – Oct 2026 |
| Labels | `category` (8), `department` (8), `priority` (3), `sla_hours` |
| Metadata | `complaint_id`, `district`, `water_body`, `channel`, `reported_at`, `split` |

The dataset is **synthetic and fully reproducible** (`python data/generate_dataset.py`). It is built from category-specific complaint templates with realistic noise (typos, all-lowercase text, Tanglish fillers, missing punctuation, and distracting side-mentions of other problems). The **test split uses sentence templates never seen in training**, so the scores measure generalisation, not memorisation.

### Categories → departments
| Category | Routed to |
|---|---|
| Industrial Effluent Discharge | State Pollution Control Board, Industrial Monitoring Cell |
| Sewage & Drainage Overflow | Municipal Corporation, Sewerage & Drainage Department |
| Drinking Water Contamination | Water Supply & Drainage Board, Water Quality Lab |
| Solid Waste & Plastic Dumping | Municipal Solid Waste Management Department |
| Agricultural Runoff | Agriculture & Water Resources Joint Cell |
| Oil & Chemical Spill | District Disaster Management Authority, Hazmat Response |
| Algal Bloom & Fish Kill | Fisheries & Aquatic Ecology Department |
| Groundwater Contamination | Groundwater Authority, Hydrogeology Wing |

## Dashboard
* **Live Classifier:** type a complaint and get a routed ticket card (category, department, priority, SLA, confidence) with a one-line LLM explanation and a probability chart.
* **Overview:** KPI cards, plus complaints by category, district, month, channel and water body.
* **Routing Queue:** filterable, SLA-sorted ticket table and department workload.
* **Model Performance:** LLM vs TF-IDF baseline, confusion matrix, per-class F1, confidence distribution.

## Run it
1. Click **Open in Colab** above.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. The dashboard opens inline, and a public `gradio.live` link is printed.

It also runs on CPU, using Qwen2.5-0.5B and a 150-complaint evaluation sample.

## Repository structure
```
├── LLM_Water_Pollution_Complaint_Routing.ipynb   # main Colab notebook (pipeline + dashboard)
├── data/
│   ├── water_pollution_complaints.csv            # AquaGrievance-TN dataset
│   └── generate_dataset.py                       # reproducible dataset generator
├── build_notebook.py                             # regenerates the notebook from source
└── requirements.txt
```

## Tech stack
Python · Hugging Face Transformers · Qwen2.5-Instruct · PyTorch · scikit-learn · Plotly · Gradio · Google Colab
