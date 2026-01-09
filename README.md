# 🍎 MonaOps: Apple Support Billing Friction Analyzer

## 🚀 Project Overview
MonaOps is an AI-driven Product Operations engine that transforms unstructured social media data into strategic product roadmaps. This specific project analyzes **Apple Support** customer interactions to determine if technical investment in the "Billing Module" should be prioritized over support-led mitigation.

## 🧠 The PM Strategy (Business Value)
- **Objective:** Identify whether recurring billing complaints are "System Failures" (Engineering fix) or "Confusing UX" (Product/Design fix).
- **Core Methodology:** Applied a deterministic "Most Likely Friction" rubric to a sample of 800+ high-intent billing tweets.
- **Key Outcome:** An automated Executive Dashboard providing an "Invest / Don't Invest" recommendation based on real-world customer pain.

## 🛠️ Key Features
- **Big Data Pipeline:** Scans 2.8M rows using Pandas chunking for high-performance memory management.
- **Deterministic AI:** Leverages **Gemini 2.0 Flash** with `temperature: 0` to ensure objective, repeatable analysis.
- **Confidence Scoring:** Displays visual confidence indicators (▲) for every metric to guide stakeholder risk assessment.
- **Resilient Engineering:** Implements automatic exponential backoff to handle API rate limits (429 errors).

## 📦 Tech Stack
- **Language:** Python 3.10+
- **AI Engine:** Google Gemini 2.0 Flash API
- **Data:** Pandas (Regex-based pattern matching)
- **Reporting:** Dynamic HTML/CSS with Chart.js visualizations

## 🚀 How to Run
1. **Clone the repository:**
   `git clone https://github.com/monaops/monaops-apple-billing.git`
2. **Install dependencies:**
   `pip install google-genai pandas python-dotenv`
3. **Configure your API Key:**
   Create a `.env` file and add: `GEMINI_API_KEY=your_actual_key_here`
4. **Execute the analysis:**
   `python scripts/advancedanalysis.py`

## 📊 Sample Dashboard

---
