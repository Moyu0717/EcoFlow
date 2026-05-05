# 🌿 EcoFlow — Smart Green Commute AI

> **MyAI Future Hackathon 2026 — Track 4: Green Horizon (Smart Cities & Mobility)**
> Team: **Can Win Just Enough** | Organised by GDG On Campus UTM

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Cloud%20Run-4285F4?style=for-the-badge&logo=google-cloud)](https://ecoflow-ai-196537430669.asia-southeast1.run.app)
[![Track](https://img.shields.io/badge/Track-Green%20Horizon-22C55E?style=for-the-badge)]()
[![Built with](https://img.shields.io/badge/Built%20with-Google%20AI-EA4335?style=for-the-badge&logo=google)]()

---

## 🚨 Problem Statement

Urban centres in Malaysia — particularly along the **Johor-Singapore Innovation Corridor** — face severe mobility congestion. The average KL commuter spends **150+ hours per year** stuck in traffic, generating unnecessary carbon emissions that push Malaysia further from its **Net Zero 2050** target.

There is no single intelligent tool that helps everyday Malaysians compare transport modes by **time, cost (RM), and carbon footprint simultaneously** — and then takes autonomous action to plan, match carpools, and ground recommendations in official national policy.

---

## 💡 Solution: EcoFlow

EcoFlow is a **dual-mode Agentic AI green mobility platform** — the only project at this hackathon that addresses Malaysia's mobility carbon problem from **both ends** of the source:

**🧍 Citizen Mode** — helps individuals choose greener routes for their daily commute
**🏙️ Planner Mode** — helps merchants, councils and developers reduce *unnecessary long-distance trips* through better local-amenity placement

EcoFlow autonomously:

1. **Plans** the most eco-friendly commute route — Drive, Carpool, Motorcycle, Grab, Bus / RapidKL, MRT / LRT, Park & Ride, **Park & Walk**, Cycling, Walking
2. **Calculates** real Malaysian costs (RM), CO₂ emissions (kg), and congestion levels
3. **Watches** every saved schedule and **proactively pings** the user 30–60 minutes before departure with weather + traffic-aware recommendations *(this is the "Chat → Action" pillar of the technical mandate)*
4. **Matches** OKU-friendly carpool partners via real accessibility metadata (Persons with Disabilities Act 2008 alignment)
5. **Verifies** Mikro enterprises with Gemini Vision reading the SSM business cert in 30 seconds (SME Corp Malaysia + SSM integration — *"Sovereign Technology Builders"*)
6. **Grounds** every recommendation in Malaysia's official transport policy via RAG (NETR, RapidKL data) — with **inline citations**
7. **Tracks** personal and community carbon savings; writes anonymised, k≥5 aggregates for Planner Mode insights (PDPA 2010 / 2024 Amendment compliant)

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     index.html (Frontend)                      │
│      Firebase Auth · Leaflet Map · Mobile-responsive UI        │
│      Citizen / Planner mode toggle · Today's Schedule card     │
│      Proactive Toast (Chat → Action) · Citation chips          │
└──────────────────────┬────────────────────────────────────────┘
                       │ HTTPS + Firebase ID-token verification
┌──────────────────────▼────────────────────────────────────────┐
│              FastAPI Backend (Google Cloud Run)                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │       EcoFlow Agentic Layer (agent.py · 9 tools)         │ │
│  │   Firebase Genkit @flow ←→ Gemini Function-calling       │ │
│  │     ├── plan_commute                                     │ │
│  │     ├── find_carpool_matches  (OKU-aware)                │ │
│  │     ├── register_carpool_offer (OKU metadata)            │ │
│  │     ├── search_malaysia_policy (RAG + citations)         │ │
│  │     ├── get_user_impact                                  │ │
│  │     ├── check_schedule_now                               │ │
│  │     ├── search_places_by_intent                          │ │
│  │     ├── analyze_site_potential (Planner)                 │ │
│  │     └── optimise_multi_stop                              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │   Proactive Agent (schedules.py)                         │ │
│  │   Cloud Scheduler → /api/v1/schedules/proactive-check    │ │
│  │   (runs every 15 min) → checks weather + traffic +       │ │
│  │   user preferences → writes Firestore notification        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │   Planner Mode (planner.py)                              │ │
│  │   /api/v1/planner/analyze-site   — site-score 0-100      │ │
│  │   /api/v1/planner/heatmap        — k-anonymous trips     │ │
│  │   /api/v1/planner/merchant/verify-mikro                  │ │
│  │       ↑ Gemini Vision reads SSM cert in 30 seconds       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌────────────┐ ┌────────────────┐ ┌──────────────────────┐  │
│  │ Gemini 2.5 │ │ Vertex AI      │ │  Firebase            │  │
│  │ Flash      │ │ Search RAG     │ │  Auth + Firestore    │  │
│  │ (text+vis) │ │ Datastore      │ │                      │  │
│  └────────────┘ └────────────────┘ └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## ✅ Google AI Ecosystem Stack (Technical Mandate)

| Requirement | Implementation | Status |
|---|---|---|
| **1. Intelligence (Brain)** | Gemini 2.5 Flash text **and Vision** via `google-generativeai` | ✅ |
| **2. Orchestrator** | Firebase Genkit `@ai.flow()` + `@ai.tool()` in `agent.py` *(per FAQ: pick one of Genkit / Vertex Agent Builder — we pick Genkit for tighter Firebase integration)* | ✅ |
| **3. Deployment Lifecycle** | Deployed on Google Cloud Run (serverless container) | ✅ |
| **4. Context (RAG)** | Vertex AI Search — `ecoflow_1776621221780` datastore (NETR + transport policy) **with inline citations** | ✅ |
| **5. Autonomous Execution** | Cloud Scheduler → Proactive Agent flow (`/api/v1/schedules/proactive-check`) — the literal "Chat → Action" pillar | ✅ |
| **Mandatory tooling** | Google AI Studio + Antigravity used for model testing and prompt iteration | ✅ |

---

## 🚀 Features

### 🧍 Citizen Mode

#### 🗺️ Smart Route Planning
- Input: origin + destination on interactive Leaflet map (or saved schedule)
- Output: All 10 transport modes ranked by personalised Eco Score (0–100)
- Real Malaysian data: RapidKL fares, Grab surge pricing, petrol costs (RM 2.05/L), MoT 2023 emission factors
- KL traffic patterns: morning rush (7–9am), evening peak (5–7pm)
- **NEW:** Park & Walk mode for short city-centre commutes (4–12 km)

#### 📅 Schedule + Proactive Agent
- Save commute schedules (one-off / weekdays / daily / custom days)
- Cloud Scheduler runs `/api/v1/schedules/proactive-check` every 15 minutes
- For each schedule firing within 15–60 min: re-computes route given **current** weather + traffic, scores by eco-score, decides if it's worth pinging
- Pushes notification message *generated by Gemini* to the user — *"Heads up — KESAS is heavy and rain expected. Take MRT, leave at 7:10 — saves 28 min and 0.8 kg CO₂."*

#### 🤝 OKU-Friendly Carpool Matching
- Carpool providers self-declare OKU friendliness, wheelchair capacity, ramp availability — these **materially affect matching results**, not just decorative tags
- OKU users prioritised (or strictly filtered) onto OKU-friendly providers
- Persons with Disabilities Act 2008 + Malaysia Madani inclusivity alignment

#### 📊 Personal Impact Tracking
- Personal CO₂ saved, RM saved, trees equivalent
- Community leaderboard, achievement badges
- Yearly projection based on commute habits

### 🏙️ Planner Mode (B2B / B2G)

#### 🎯 Site Analysis — three views, one analysis
- **Merchant view**: "Should I open here?" — score 0-100 + foot traffic + competition + estimated CO₂ saved/month if a missing amenity is added
- **Resident / Council view**: "What's this neighbourhood missing?" — same analysis, framed as a *carbon-saving opportunity* (15-Minute City logic)
- **Developer view**: "Will this development reduce or increase car dependence?" — qualitative carbon-impact preview

#### 🪪 Mikro Enterprise Verification (Gemini Vision)
- Mikro = annual revenue < RM 300k OR < 5 employees (per SME Corp Malaysia)
- User uploads a photo of their **SSM business registration certificate**
- **Gemini 2.5 Flash Vision** extracts: SSM number, company name, registration date — in 30 seconds
- Verified Mikro accounts get Planner Mode at zero cost — *"Sovereign Technology Builders"* in practice (we use Malaysia's own digital infrastructure, not parallel KYC)

### 🛡️ Privacy Architecture (PDPA 2010 + 2024 Amendment)
1. **Opt-in consent** — users explicitly choose to contribute anonymised aggregates to Planner analytics
2. **K-anonymity** — any insight derived from < 5 users is suppressed
3. **Data minimisation** — analytics database stores no `user_id`; trips are bucketed to ~100 m grid cells
4. **Insights, not data** — merchants receive AI-generated reports, never raw user lists
5. Same approach as **Google Maps Popular Times** and **Strava Metro for Cities**

### 🤖 Agentic AI Chat
- Autonomous multi-step reasoning: intent → tool selection → execution → grounded response
- 9 callable tools, up to 5 chained calls per query
- Integrated with Gemini function-calling for dynamic tool orchestration
- Grounded in Malaysia's NETR policy via Vertex AI Search — **with inline citations rendered as chips**
- Graceful fallback to raw Gemini when Genkit is unavailable

### 🌤️ Live Conditions
- Weather via Open-Meteo API (no key required)
- Real-time congestion factor by time-of-day

### 📱 Mobile-First UI (NEW)
- Full responsive layout for screens ≥ 360 px wide
- Side panels become bottom sheets on mobile
- Bottom tab bar replaces desktop nav rail
- 44 px minimum tap targets, 16 px input fonts (no iOS auto-zoom)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JS, Leaflet.js, mobile-responsive |
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI Brain | Google Gemini 2.5 Flash — text and **multimodal Vision** (`google-generativeai`) |
| AI Orchestration | Firebase Genkit (`genkit`) — single-agent architecture |
| RAG / Knowledge | Vertex AI Search (`google-cloud-discoveryengine`) — datastore `ecoflow_1776621221780` |
| Auth & Database | Firebase Auth + Firestore (`firebase-admin`) — real ID-token verification |
| Scheduling | Cloud Scheduler — calls `/api/v1/schedules/proactive-check` every 15 min |
| Deployment | Google Cloud Run (asia-southeast1) |
| Routing | OSRM (Open Source Routing Machine) |
| Maps | Leaflet.js + OpenStreetMap |
| File upload | `python-multipart` for SSM cert verification |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- A Google Cloud project with Vertex AI Search enabled
- Firebase project with Auth + Firestore
- Gemini API key from Google AI Studio

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecoflow-ai.git
cd ecoflow-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
FIREBASE_KEY_PATH=firebase-key.json
GCP_PROJECT_ID=my-future-ai-493816
GCP_LOCATION=global
GCP_DATASTORE_ID=ecoflow_1776621221780
MAPBOX_TOKEN=your_mapbox_token_here   # optional
```

### 4. Add Firebase service account key
Download your Firebase Admin SDK key and save as `firebase-key.json` in the root directory.

### 5. Run the backend
```bash
uvicorn main:app --reload --port 8080
```

### 6. Open the frontend
Visit `http://localhost:8080` in your browser.

---

## 🌐 Live Deployment

**Cloud Run URL:** https://ecoflow-ai-196537430669.asia-southeast1.run.app

## 🔌 API Endpoints (Production – Cloud Run)
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main app (index.html) |
| `/health` | GET | Health check |
| `/api/v1/full-analysis` | POST | AI route recommendation |
| `/api/v1/agent` | POST | Agentic AI chat (Genkit + Gemini tools) |
| `/api/v1/ai-chat` | POST | RAG-grounded chat (Vertex AI Search) |
| `/api/v1/smart-routing` | POST | Transport options |
| `/api/v1/save-trip` | POST | Save completed trip |
| `/api/v1/impact/{user_id}` | GET | Personal carbon impact |
| `/api/v1/community-impact` | GET | Global stats |
| `/api/v1/leaderboard` | GET | Top eco-commuters |
| `/api/v1/carpool-match` | POST | Find carpool partners |
| `/docs` | GET | Interactive API docs (Swagger) |

---

## 🤖 AI Disclosure

This project used the following AI coding tools during development:
- **Google Gemini** (via AI Studio) — for code generation assistance and prompt engineering
- **Firebase Genkit** — as the agentic orchestration framework

All AI-generated code has been reviewed, tested, and understood by the team. Every part of the codebase can be explained and defended by team members during judging.

---

## 🌱 Impact & Malaysian Context

EcoFlow directly addresses **Track 4: Green Horizon** by:

- Helping Malaysians make data-driven commute decisions grounded in real RM costs
- Supporting Malaysia's **Net Zero 2050** target through behavioural carbon tracking
- Promoting MRT/LRT adoption along the **Johor-Singapore Innovation Corridor**
- Reducing per-capita carbon footprint through intelligent carpool matching
- Grounding all recommendations in Malaysia's **National Energy Transition Roadmap (NETR)**

---

## 👥 Team — Can Win Just Enough

Built with 💚 for the MyAI Future Hackathon 2026
Organised by Google Developer Groups On Campus UTM

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
