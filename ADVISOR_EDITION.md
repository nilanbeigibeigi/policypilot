# PolicyPilot — Advisor Edition

The same RAG engine, pointed at a **fictional financial advisory firm** (Brightwater Wealth Partners) instead of the grocery store.

`docs_advisor/` holds 6 synthetic documents an advisory office looks things up in every day:

| Document | What staff ask it |
|---|---|
| Client Onboarding Procedure | Documents needed, KYC updates, risk profiles, account-opening timeline |
| Client Review Process | Review frequency, prep checklist, life-event triggers, note-taking rules |
| Group Benefits Booklet (fictional client company) | Eligibility, dental, paramedical, LTD, claims |
| Life Insurance Policy Summary (sample client) | Beneficiaries, conversion, grace period, exclusions |
| Privacy, Data and AI Use Policy | Secure sending, AI tool rules, retention, breaches |
| Complaints and Compliance | Complaint timelines, trade errors, gifts, outside activities |

All names and numbers are synthetic. Nothing here is a real contract or a real firm's policy.

## Run it

PowerShell (Windows):
```powershell
$env:POLICYPILOT_DOCS="docs_advisor"; $env:ORG_NAME="Brightwater Wealth Partners"; $env:AUDIENCE="advisors and staff"
python -m streamlit run app.py
```
macOS / Linux:
```bash
POLICYPILOT_DOCS=docs_advisor ORG_NAME="Brightwater Wealth Partners" AUDIENCE="advisors and staff" python -m streamlit run app.py
```

## Evaluate it
```powershell
$env:POLICYPILOT_DOCS="docs_advisor"; $env:EVAL_FILE="eval_questions_advisor.json"; python eval.py
```
24 questions: 20 answerable + 4 that must be refused (portfolio return, commission, Wi-Fi password, stock picks).
Keyword (TF-IDF) baseline retrieval: 100% at top-1 and top-3. Run the command above for the semantic-search result.
