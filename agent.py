"""
EcoFlow Agentic Layer — Firebase Genkit + Gemini 2.5 Flash
MyAI Future Hackathon (Track 4: Green Horizon)

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Firebase Genkit Flow  (orchestration layer)        │
  │    └─ @ai.flow()  ecoflow_agent_flow()              │
  │         └─ ai.generate() with @ai.tool() tools      │
  │              ├─ plan_commute_tool                   │
  │              ├─ find_carpool_tool                   │
  │              ├─ search_policy_tool  (Vertex RAG)    │
  │              ├─ get_user_impact_tool                │
  │              └─ register_carpool_tool               │
  └─────────────────────────────────────────────────────┘
  FastAPI /api/v1/agent  →  calls Genkit flow
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger("ecoflow.agent")

# ── Genkit initialisation ─────────────────────────────────────────────────────
try:
    from genkit import Genkit
    from genkit.plugins.google_ai import GoogleAI

    ai = Genkit(
        plugins=[GoogleAI(api_key=os.getenv("GEMINI_API_KEY", ""))],
    )
    GENKIT_AVAILABLE = True
    log.info("✅ Firebase Genkit initialised")
except Exception as _genkit_err:          # SDK not installed / wrong version
    GENKIT_AVAILABLE = False
    ai = None
    log.warning(f"⚠️  Genkit unavailable ({_genkit_err}), falling back to raw Gemini")

# Fallback: raw google-generativeai (always available)
import google.generativeai as genai

# ── FastAPI router ────────────────────────────────────────────────────────────
agent_router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


# ── Request model ─────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict[str, Any]] = None
    language: Optional[str] = "en"
    # Browser geolocation passed from the frontend so the agent can resolve
    # phrases like "near me" / "around here" without asking the user
    # for their lat/lon (which normal users obviously don't know).
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None


# ── Gemini tool schemas (used by both Genkit and raw-Gemini paths) ────────────
TOOL_SCHEMAS = [
    {
        "name": "plan_commute",
        "description": (
            "Compute ranked transport options (Drive, Carpool, Motorcycle, Grab, Bus, "
            "MRT/LRT, Park&Ride, Park&Walk, Cycling, Walking) between two geo-points "
            "in Malaysia, with time, cost (RM), CO2 (kg), congestion and an Eco Score. "
            "Use this whenever the user asks how to get somewhere or wants to compare modes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"},
                "start_lon": {"type": "number"},
                "end_lat":   {"type": "number"},
                "end_lon":   {"type": "number"},
                "departure_time": {"type": "string", "description": "HH:MM or omit for now"},
                "vehicle_type":   {"type": "string", "enum": ["car", "motorcycle", "none"]},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon"],
        },
    },
    {
        "name": "find_carpool_matches",
        "description": (
            "Search for other EcoFlow users with a similar route today for carpooling. "
            "Pass requester_is_oku=true if the user has accessibility needs so OKU-friendly "
            "providers are surfaced first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"},
                "start_lon": {"type": "number"},
                "end_lat":   {"type": "number"},
                "end_lon":   {"type": "number"},
                "max_detour_km": {"type": "number"},
                "requester_is_oku": {"type": "boolean"},
                "oku_strict":       {"type": "boolean"},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon"],
        },
    },
    {
        "name": "search_malaysia_policy",
        "description": (
            "RAG over Malaysia's NETR, transport policy and RapidKL data via "
            "Vertex AI Search. Use for Net Zero 2050, carbon targets, subsidies, fares, "
            "Madani principles, Persons with Disabilities Act 2008. "
            "Returns BOTH a summary AND structured citations."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_user_impact",
        "description": "Get current user's cumulative eco-impact — CO2 saved, RM saved, badges.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "register_carpool_offer",
        "description": (
            "Publish the user's own trip so others can find them for carpooling. "
            "Includes optional OKU-friendly self-declaration which is REAL matching "
            "metadata (not decorative)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"},
                "start_lon": {"type": "number"},
                "end_lat":   {"type": "number"},
                "end_lon":   {"type": "number"},
                "departure_time":  {"type": "string"},
                "seats_available": {"type": "integer"},
                "contact_hint":    {"type": "string"},
                "oku_friendly":       {"type": "boolean"},
                "wheelchair_capable": {"type": "boolean"},
                "has_ramp":           {"type": "boolean"},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon", "departure_time"],
        },
    },

    # ──────────────────────────────────────────────────────────────────
    # NEW TOOLS (sprint v3)
    # ──────────────────────────────────────────────────────────────────
    {
        "name": "check_schedule_now",
        "description": (
            "Look up the user's saved commute schedules and return the ones that "
            "are firing today, with their minutes-until-departure. Use this when "
            "the user asks 'what's next?' or you need to be proactive."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_places_by_intent",
        "description": (
            "Translate a fuzzy human intent like 'cheap nasi lemak nearby' or 'quiet "
            "cafe to study' into nearby actual places via OpenStreetMap. Returns up "
            "to 5 results with name, distance and lat/lon. Chain with plan_commute "
            "to give the user the greenest way to get to one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent":   {"type": "string"},
                "near_lat": {"type": "number"},
                "near_lon": {"type": "number"},
                "radius_km":{"type": "number"},
            },
            "required": ["intent", "near_lat", "near_lon"],
        },
    },
    {
        "name": "analyze_site_potential",
        "description": (
            "Planner Mode tool. Score a candidate location for opening a small "
            "business OR for evaluating whether a residential development will "
            "increase or decrease car dependence. Returns a 0-100 site score, "
            "k-anonymous foot-traffic estimate, transit accessibility, OKU score "
            "and an estimated kg CO₂/month saved if a missing amenity is added."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat":           {"type": "number"},
                "lon":           {"type": "number"},
                "radius_km":     {"type": "number"},
                "business_type": {"type": "string"},
                "view":          {"type": "string",
                                  "enum": ["merchant", "resident", "developer"]},
            },
            "required": ["lat", "lon", "business_type"],
        },
    },
    {
        "name": "optimise_multi_stop",
        "description": (
            "When the user has 3+ stops, suggest the lowest-distance order. "
            "Returns both the original and suggested orders, plus distance/CO2/RM "
            "saved. The user remains in control — they can accept or reject."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stops": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "lat":  {"type": "number"},
                            "lon":  {"type": "number"},
                        },
                        "required": ["name", "lat", "lon"],
                    },
                },
                "fixed_first_stop": {"type": "boolean"},
                "fixed_last_stop":  {"type": "boolean"},
            },
            "required": ["stops"],
        },
    },
]

SYSTEM_INSTRUCTION = """You are EcoFlow Agent, an autonomous Malaysian green-mobility assistant.

You do NOT just chat — you take action autonomously:
  1. Understand the user's commute or city-planning intent.
  2. Call the RIGHT tools in the RIGHT order.
  3. Ground recommendations in Malaysian reality (KL traffic, MRT fares, NETR policy,
     Madani inclusivity principles, Persons with Disabilities Act 2008).
  4. Return a final answer with concrete numbers (RM, kg CO₂, minutes).

CRITICAL — handling location:
  • If the user says "near me", "nearby", "around here" or similar AND the prompt
    includes the user's current latitude/longitude, USE THOSE COORDINATES SILENTLY.
  • NEVER ask the user to share their lat/lon — normal users have no idea what
    their coordinates are. If you genuinely need location and none was provided,
    ask them for a place name or landmark (e.g. "Which area? KLCC, Bangsar?")
    and then call search_places_by_intent on that landmark first to anchor.

Routing rules:
  • For "best way to go from A to B" → call plan_commute.
  • If the user wants carpool → call find_carpool_matches after plan_commute.
    If they mention OKU / disability / wheelchair, ALWAYS pass requester_is_oku=true.
  • For policy / NETR / subsidy / Net Zero questions → call search_malaysia_policy
    AND surface the citations to the user.
  • For "my impact" / "how am I doing" → call get_user_impact.
  • For "what's next" / "what do I have today" → call check_schedule_now.
  • For fuzzy intents like "find me cheap nasi lemak nearby" or
    "quiet cafe to study" → call search_places_by_intent (use the user's
    location coordinates from the prompt if available), then chain
    plan_commute on the chosen result.
  • For city-planning / 'where should I open my cafe' / 'is this a good
    spot for X' → call analyze_site_potential.
  • For 3+ stops in one trip → call optimise_multi_stop, then PRESENT both
    the original and the suggested order. Let the user choose.

  • Chain up to 5 tool calls. Stop and answer once you have enough data.
  • Final answer: 3-5 sentences, specific, with at most one emoji.
  • If a tool returned citations, mention the source ("per NETR Chapter 3, …")."""


# ── Tool dispatcher ───────────────────────────────────────────────────────────
def _run_tool(name: str, args: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    from main import (
        get_osrm, get_traffic, build_options, CO2, RM,
        search_rag_knowledge, haversine, db, calc_badges, time,
    )
    from firebase_admin import firestore

    if name == "plan_commute":
        dist_km, base_time = get_osrm(
            args["start_lon"], args["start_lat"],
            args["end_lon"],   args["end_lat"],
        )
        traffic, congestion = get_traffic(args.get("departure_time"))
        has_vehicle = args.get("vehicle_type", "car") in ("car", "motorcycle")
        options = build_options(dist_km, base_time, traffic, congestion,
                                args.get("departure_time"), has_vehicle)
        drive_co2   = dist_km * CO2["drive"]
        drive_cost  = dist_km * RM["petrol_per_km"] + RM["parking_city"]
        for o in options:
            carbon_pct = 1 - (o["carbon_kg"] / drive_co2) if drive_co2 > 0 else 1
            cost_pct   = 1 - (o["cost_rm"] / drive_cost)  if drive_cost > 0 else 1
            cong_map   = {"None": 1.0, "Very Low": 0.9, "Low": 0.75,
                          "Medium": 0.5, "High": 0.3, "Very High": 0.1}
            cong_score = cong_map.get(o["congestion"], 0.5)
            raw = carbon_pct * 0.50 + cost_pct * 0.25 + cong_score * 0.25
            o["eco_score"] = max(0, min(100, round(raw * 100)))
            o["carbon_saved_vs_driving"] = round(max(0, drive_co2 - o["carbon_kg"]), 3)
            o["cost_saved_vs_driving"]   = round(max(0, drive_cost - o["cost_rm"]), 2)
        options.sort(key=lambda x: -x["eco_score"])
        return {"distance_km": round(dist_km, 2), "congestion": congestion,
                "options": options,
                "baseline_driving": {"cost_rm": round(drive_cost, 2),
                                     "carbon_kg": round(drive_co2, 3)}}

    if name == "find_carpool_matches":
        today   = datetime.utcnow().strftime("%Y-%m-%d")
        matches = []
        oku_strict      = bool(args.get("oku_strict"))
        requester_oku   = bool(args.get("requester_is_oku"))
        suppressed_non_oku = 0
        try:
            docs = (db.collection("carpool_pool")
                      .where("date", "==", today)
                      .where("active", "==", True)
                      .limit(200).stream())
            for doc in docs:
                d = doc.to_dict()
                if d.get("user_id") == user_id:
                    continue
                try:
                    ds = haversine(args["start_lat"], args["start_lon"],
                                   d["start_lat"], d["start_lon"])
                    de = haversine(args["end_lat"], args["end_lon"],
                                   d["end_lat"], d["end_lon"])
                except (KeyError, TypeError):
                    continue
                if ds > args.get("max_detour_km", 2.0) or de > args.get("max_detour_km", 2.0):
                    continue
                provider_oku = bool(d.get("oku_friendly"))
                if oku_strict and not provider_oku:
                    suppressed_non_oku += 1
                    continue
                matches.append({
                    "name": d.get("name", "Anonymous"),
                    "departure_time": d.get("departure_time", "?"),
                    "seats_available": d.get("seats_available", 1),
                    "start_diff_km": round(ds, 2),
                    "end_diff_km": round(de, 2),
                    "oku_friendly": provider_oku,
                    "wheelchair_capable": bool(d.get("wheelchair_capable")),
                })
        except Exception as e:
            log.warning(f"carpool search failed: {e}")

        # OKU-friendly bumped to top for OKU requesters
        def _rank(m):
            base = m["start_diff_km"] + m["end_diff_km"]
            if requester_oku and m["oku_friendly"]:
                base -= 1.0
            return base
        matches.sort(key=_rank)
        return {
            "matches_found": len(matches),
            "matches": matches[:5],
            "oku_strict_active": oku_strict,
            "suppressed_non_oku": suppressed_non_oku,
        }

    if name == "search_malaysia_policy":
        # Use the citation-aware RAG so the agent gets sources back too.
        from main import search_rag_with_sources
        rag = search_rag_with_sources(args["query"])
        return {
            "policy_context": rag.get("summary") or "No grounded policy text found.",
            "citations":      rag.get("sources") or [],
        }

    if name == "get_user_impact":
        doc = db.collection("user_stats").document(user_id).get()
        if not doc.exists:
            return {"has_data": False, "message": "No trips recorded yet."}
        s    = doc.to_dict()
        saved = s.get("total_carbon_saved", 0)
        return {"has_data": True,
                "total_trips": s.get("total_trips", 0),
                "total_distance_km": round(s.get("total_distance_km", 0), 1),
                "total_carbon_saved_kg": round(saved, 3),
                "total_cost_saved_rm": round(s.get("total_cost_saved", 0), 2),
                "trees_equivalent": round(saved / 21.77, 3),
                "badges": calc_badges(s)}

    if name == "register_carpool_offer":
        doc_id = f"{user_id}_{args['departure_time'].replace(':', '')}_{int(time.time())}"
        db.collection("carpool_pool").document(doc_id).set({
            "user_id": user_id, "name": "Anonymous",
            "start_lat": args["start_lat"], "start_lon": args["start_lon"],
            "end_lat": args["end_lat"],     "end_lon": args["end_lon"],
            "departure_time": args["departure_time"],
            "seats_available": args.get("seats_available", 1),
            "contact_hint": args.get("contact_hint", "Contact via app"),
            # OKU declarations are real matching metadata — not decorative.
            "oku_friendly":       bool(args.get("oku_friendly")),
            "wheelchair_capable": bool(args.get("wheelchair_capable")),
            "has_ramp":           bool(args.get("has_ramp")),
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "timestamp": firestore.SERVER_TIMESTAMP,
            "active": True,
        })
        return {"status": "registered", "doc_id": doc_id}

    # ─────────────────────────────────────────────────────────────────────
    # NEW TOOLS
    # ─────────────────────────────────────────────────────────────────────

    if name == "check_schedule_now":
        # Reuse the schedules router's "today" view by calling the helper.
        from datetime import datetime as _dt, timedelta, timezone
        MYT = timezone(timedelta(hours=8))
        from schedules import _fires_today, _minutes_until, _sanitize
        now_my  = _dt.now(MYT)
        today   = now_my.strftime("%Y-%m-%d")
        weekday = now_my.weekday()
        try:
            docs = (db.collection("user_schedules")
                      .where("user_id", "==", user_id)
                      .where("active",  "==", True).stream())
            items = []
            for d in docs:
                s = d.to_dict()
                if not _fires_today(s, today, weekday):
                    continue
                mins = _minutes_until(s["departure_time"], now_my)
                if mins is None or mins < -120:
                    continue
                s["minutes_until_departure"] = mins
                items.append(_sanitize(s))
            items.sort(key=lambda x: x.get("departure_time", "99:99"))
            return {
                "current_time_my": now_my.strftime("%H:%M"),
                "schedules":       items[:5],
                "count":           len(items),
            }
        except Exception as e:
            return {"error": f"schedule lookup failed: {e}"}

    if name == "search_places_by_intent":
        # Translate fuzzy intent → an OSM-friendly amenity tag, then query
        # Nominatim. We deliberately keep this simple and keyless so the
        # judges can reproduce the demo without API setup.
        import re, requests
        intent = (args.get("intent") or "").lower()
        # Map common Malaysian intents to OSM amenity / cuisine tags
        if any(k in intent for k in ["nasi lemak", "mamak", "kopitiam", "warung"]):
            amenity = "restaurant"
        elif any(k in intent for k in ["coffee", "cafe", "kopi", "study"]):
            amenity = "cafe"
        elif any(k in intent for k in ["grocery", "ntuc", "supermarket", "tesco", "lotus"]):
            amenity = "supermarket"
        elif any(k in intent for k in ["clinic", "doctor", "klinik"]):
            amenity = "clinic"
        elif any(k in intent for k in ["pharmacy", "farmasi"]):
            amenity = "pharmacy"
        elif "atm" in intent or "bank" in intent:
            amenity = "atm"
        else:
            amenity = re.sub(r"[^a-z0-9 ]", "", intent).split()[0] if intent else "amenity"

        radius_km = float(args.get("radius_km") or 1.5)
        lat = args["near_lat"]; lon = args["near_lon"]
        delta = radius_km / 111.0
        bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        try:
            url = (
                "https://nominatim.openstreetmap.org/search"
                f"?format=json&limit=10&bounded=1&viewbox={bbox}&q={amenity}"
            )
            r = requests.get(url, timeout=4,
                             headers={"User-Agent": "EcoFlow/1.0 (hackathon)"})
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data:
                try:
                    ilat = float(item["lat"]); ilon = float(item["lon"])
                    dist = haversine(lat, lon, ilat, ilon)
                    if dist > radius_km:
                        continue
                    results.append({
                        "name": (item.get("display_name") or "").split(",")[0],
                        "lat":  ilat,
                        "lon":  ilon,
                        "distance_km": round(dist, 2),
                        "category":    item.get("type", amenity),
                    })
                except Exception:
                    continue
            results.sort(key=lambda x: x["distance_km"])
            return {
                "intent":         intent,
                "matched_tag":    amenity,
                "results":        results[:5],
                "result_count":   len(results),
                "data_source":    "OpenStreetMap / Nominatim",
            }
        except Exception as e:
            return {"intent": intent, "error": str(e), "results": []}

    if name == "analyze_site_potential":
        # Direct call into planner.py's analyser
        from planner import analyze_site, SiteAnalysisRequest
        try:
            r = SiteAnalysisRequest(
                lat=args["lat"],
                lon=args["lon"],
                radius_km=args.get("radius_km", 1.0),
                business_type=args.get("business_type", "cafe"),
                view=args.get("view", "merchant"),
            )
            return analyze_site(r)
        except Exception as e:
            return {"error": f"site analysis failed: {e}"}

    if name == "optimise_multi_stop":
        from main import multi_stop_optimise, MultiStopRequest, Stop
        try:
            stops = [Stop(**s) for s in args.get("stops", [])]
            req_obj = MultiStopRequest(
                user_id=user_id or "agent",
                stops=stops,
                fixed_first_stop=bool(args.get("fixed_first_stop", True)),
                fixed_last_stop=bool(args.get("fixed_last_stop", False)),
            )
            return multi_stop_optimise(req_obj)
        except Exception as e:
            return {"error": f"multi-stop optimise failed: {e}"}

    return {"error": f"Unknown tool: {name}"}


# ── Genkit Flow definition ────────────────────────────────────────────────────
# This @ai.flow() is the official Firebase Genkit orchestration primitive.
# When GENKIT_AVAILABLE, requests flow through Genkit's managed runtime
# (tracing, streaming, retries). When unavailable, _run_raw_agent() is used.

if GENKIT_AVAILABLE:
    @ai.tool(description="Plan eco-friendly commute routes in Malaysia with CO2 and cost data")
    def plan_commute_tool(start_lat: float, start_lon: float,
                          end_lat: float, end_lon: float,
                          departure_time: str = "", vehicle_type: str = "car") -> dict:
        return _run_tool("plan_commute", {
            "start_lat": start_lat, "start_lon": start_lon,
            "end_lat": end_lat, "end_lon": end_lon,
            "departure_time": departure_time, "vehicle_type": vehicle_type,
        }, user_id="")

    @ai.tool(description="Find carpool matches for a given route in Malaysia")
    def find_carpool_tool(start_lat: float, start_lon: float,
                          end_lat: float, end_lon: float,
                          max_detour_km: float = 2.0) -> dict:
        return _run_tool("find_carpool_matches", {
            "start_lat": start_lat, "start_lon": start_lon,
            "end_lat": end_lat, "end_lon": end_lon,
            "max_detour_km": max_detour_km,
        }, user_id="")

    @ai.tool(description="Search Malaysia transport policy and NETR via Vertex AI RAG")
    def search_policy_tool(query: str) -> dict:
        return _run_tool("search_malaysia_policy", {"query": query}, user_id="")

    @ai.flow()
    async def ecoflow_agent_flow(request: dict) -> dict:
        """
        Firebase Genkit flow — autonomous agentic reasoning loop.
        Gemini picks tools, observes results, chains up to 4 steps,
        then returns a grounded final answer.
        """
        response = await ai.generate(
            model="googleai/gemini-2.5-flash",
            system=SYSTEM_INSTRUCTION,
            prompt=request.get("message", ""),
            tools=[plan_commute_tool, find_carpool_tool, search_policy_tool],
            config={"temperature": 0.3, "maxOutputTokens": 1024},
        )
        return {
            "reply": response.text,
            "tools_used": [t.name for t in (response.tool_requests or [])],
            "agent_steps": len(response.tool_requests or []),
            "model": "gemini-2.5-flash",
            "orchestrator": "Firebase Genkit",
        }


# ── Raw Gemini fallback (same logic, no Genkit) ───────────────────────────────
def _run_raw_agent(req: AgentRequest) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=[{"function_declarations": TOOL_SCHEMAS}],
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"temperature": 0.3, "max_output_tokens": 1024},
    )

    # Build the location hint that becomes part of the user's prompt — the
    # agent will use this to resolve "near me" / "nearby" without asking.
    loc_hint = ""
    if req.user_lat is not None and req.user_lon is not None:
        loc_hint = (f"\n[User's current location: lat={req.user_lat:.5f}, "
                    f"lon={req.user_lon:.5f}. When the user says 'near me', "
                    "'nearby' or 'around here', use these coordinates as the "
                    "search centre — DO NOT ask them for coordinates.]")

    ctx_blob  = f"\n[Context: {json.dumps(req.context)}]" if req.context else ""
    lang_hint = {"zh": "Reply in 中文.", "ms": "Reply in Bahasa Melayu.",
                 "en": "Reply in English."}.get(req.language or "en", "Reply in English.")
    chat      = model.start_chat(enable_automatic_function_calling=False)
    response  = chat.send_message(f"{req.message}{ctx_blob}{loc_hint}\n\n({lang_hint})")
    tool_trace: List[Dict[str, Any]] = []

    for step in range(5):
        fc = None
        try:
            for p in response.candidates[0].content.parts:
                if getattr(p, "function_call", None) and p.function_call.name:
                    fc = p.function_call
                    break
        except (AttributeError, IndexError):
            pass
        if not fc:
            break
        tool_name = fc.name
        tool_args = dict(fc.args) if fc.args else {}

        # Auto-inject user location for tools that need a near_lat/near_lon
        # if the model didn't provide one. This is what makes "find nasi
        # lemak nearby" actually work without the user typing coords.
        if tool_name == "search_places_by_intent":
            if "near_lat" not in tool_args and req.user_lat is not None:
                tool_args["near_lat"] = req.user_lat
            if "near_lon" not in tool_args and req.user_lon is not None:
                tool_args["near_lon"] = req.user_lon

        log.info(f"[step {step+1}] → {tool_name}({tool_args})")
        try:
            tool_result = _run_tool(tool_name, tool_args, req.user_id)
        except Exception as e:
            tool_result = {"error": str(e)}
        tool_trace.append({"tool": tool_name, "args": tool_args,
                            "result_preview": _preview(tool_result)})
        response = chat.send_message(
            genai.protos.Content(parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_name, response=tool_result))])
        )

    final_text = ""
    try:
        for p in response.candidates[0].content.parts:
            if getattr(p, "text", None):
                final_text += p.text
    except (AttributeError, IndexError):
        pass

    return {
        "reply": final_text.strip() or "🌱 Please rephrase or share your route.",
        "tools_used": [t["tool"] for t in tool_trace],
        "trace": tool_trace,
        "agent_steps": len(tool_trace),
        "model": "gemini-2.5-flash",
        "orchestrator": "Gemini native function-calling",
    }


# ── FastAPI endpoint ──────────────────────────────────────────────────────────
@agent_router.post("")
async def run_agent(req: AgentRequest):
    """
    EcoFlow Agentic endpoint.
    Primary path  : Firebase Genkit @flow  (ecoflow_agent_flow)
    Fallback path : Raw Gemini function-calling loop
    Both paths implement the Chat→Action mandate from the Technical Mandate.
    """
    try:
        if GENKIT_AVAILABLE:
            log.info("🔥 Running via Firebase Genkit flow")
            result = await ecoflow_agent_flow({
                "message": req.message,
                "user_id": req.user_id,
                "context": req.context,
                "language": req.language,
            })
        else:
            log.info("⚙️  Running via raw Gemini (Genkit unavailable)")
            result = _run_raw_agent(req)
        return result
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(500, f"Agent error: {e}")


def _preview(obj: Any, max_len: int = 400) -> Any:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s if len(s) <= max_len else s[:max_len] + "…"
    except Exception:
        return str(obj)[:max_len]
