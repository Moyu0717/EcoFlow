"""
EcoFlow Schedules + Proactive Agent
====================================
Implements the "Chat → Action (Autonomous Execution)" pillar of the
Project 2030 Technical Mandate.

Design:
  • Users create commute schedules (one-off OR recurring weekdays).
  • Cloud Scheduler hits /api/v1/schedules/proactive-check every 15 minutes.
  • For every schedule whose departure_time is 30-60 min ahead, we:
      1. Re-compute the best transport option using current traffic + weather.
      2. Compare to user's usual choice; if a better eco/time alternative
         exists, write a notification doc to Firestore.
  • The web client polls /api/v1/schedules/notifications/{user_id} every
     60 s and surfaces the toast when a fresh notification arrives.

This file is purposely self-contained — it imports helpers from main.py
(get_osrm, get_traffic, build_options, CO2, RM, db, call_gemini, log)
so we don't duplicate the route-planning math.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from firebase_admin import firestore

log = logging.getLogger("ecoflow.schedules")

# Malaysian timezone (UTC+8). Cloud Run runs in UTC, so we always
# compute "what time is it for the user?" through this offset.
MYT = timezone(timedelta(hours=8))

# ── FastAPI router ────────────────────────────────────────────────────────────
schedules_router = APIRouter(prefix="/api/v1/schedules", tags=["Schedules"])


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class ScheduleCreate(BaseModel):
    """A commute schedule a user wants EcoFlow to watch over."""
    user_id:        str
    label:          str = Field(..., max_length=60, description="e.g. 'Office', 'Class'")

    # Origin / destination
    start_lat:      float = Field(..., ge=-90, le=90)
    start_lon:      float = Field(..., ge=-180, le=180)
    end_lat:        float = Field(..., ge=-90, le=90)
    end_lon:        float = Field(..., ge=-180, le=180)
    start_name:     Optional[str] = "Home"
    end_name:       Optional[str] = "Destination"

    # Timing — local Malaysia time
    departure_time: str = Field(..., pattern=r"^\d{2}:\d{2}$",
                                description="HH:MM in Malaysia time")

    # Recurrence
    repeat:         str = Field(default="once",
                                description="'once' | 'weekdays' | 'daily' | 'custom'")
    repeat_days:    Optional[List[int]] = Field(
        default=None,
        description="0=Mon..6=Sun, only used when repeat='custom'"
    )
    one_off_date:   Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="YYYY-MM-DD, required when repeat='once'"
    )

    # Preferences (denormalized so proactive agent doesn't re-query profile)
    preferred_mode: Optional[str] = Field(
        default=None,
        description="User's usual choice: 'Drive', 'MRT / LRT', 'Bus / RapidKL', etc."
    )


class ScheduleUpdate(BaseModel):
    label:          Optional[str] = None
    departure_time: Optional[str] = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    repeat:         Optional[str] = None
    repeat_days:    Optional[List[int]] = None
    one_off_date:   Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    active:         Optional[bool] = None


# ─────────────────────────────────────────────────────────────────────────────
# CRUD endpoints
# ─────────────────────────────────────────────────────────────────────────────

@schedules_router.post("")
def create_schedule(req: ScheduleCreate):
    """Create a new commute schedule and persist it in Firestore."""
    from main import db

    # Validate recurrence rules
    if req.repeat == "once" and not req.one_off_date:
        raise HTTPException(400, "repeat='once' requires one_off_date (YYYY-MM-DD).")
    if req.repeat == "custom" and not req.repeat_days:
        raise HTTPException(400, "repeat='custom' requires repeat_days.")

    schedule_id = f"{req.user_id}_{int(time.time() * 1000)}"
    doc = {
        "schedule_id":    schedule_id,
        "user_id":        req.user_id,
        "label":          req.label,
        "start_lat":      req.start_lat,
        "start_lon":      req.start_lon,
        "end_lat":        req.end_lat,
        "end_lon":        req.end_lon,
        "start_name":     req.start_name,
        "end_name":       req.end_name,
        "departure_time": req.departure_time,
        "repeat":         req.repeat,
        "repeat_days":    req.repeat_days or [],
        "one_off_date":   req.one_off_date,
        "preferred_mode": req.preferred_mode,
        "active":         True,
        "created_at":     firestore.SERVER_TIMESTAMP,
    }
    db.collection("user_schedules").document(schedule_id).set(doc)
    log.info(f"📅 Schedule created: {schedule_id} for user {req.user_id[:6]}***")
    return {"status": "created", "schedule_id": schedule_id, "schedule": _sanitize(doc)}


@schedules_router.get("/{user_id}")
def list_schedules(user_id: str):
    """Return all active schedules for a user."""
    from main import db

    docs = (db.collection("user_schedules")
              .where("user_id", "==", user_id)
              .where("active", "==", True)
              .stream())
    items = [_sanitize(d.to_dict()) for d in docs]
    # Sort by departure_time so the UI shows earliest first
    items.sort(key=lambda x: x.get("departure_time", "99:99"))
    return {"user_id": user_id, "schedules": items, "count": len(items)}


@schedules_router.get("/{user_id}/today")
def schedules_today(user_id: str):
    """
    Return only the schedules that fire TODAY in Malaysia time.
    The web app's "Today's Schedule" card calls this on load.
    """
    from main import db

    now_my = datetime.now(MYT)
    today_iso = now_my.strftime("%Y-%m-%d")
    weekday = now_my.weekday()  # 0=Mon..6=Sun

    docs = (db.collection("user_schedules")
              .where("user_id", "==", user_id)
              .where("active", "==", True)
              .stream())

    items: List[Dict[str, Any]] = []
    for d in docs:
        s = d.to_dict()
        if not _fires_today(s, today_iso, weekday):
            continue
        # Annotate with "minutes until departure" for the UI
        mins = _minutes_until(s["departure_time"], now_my)
        if mins is not None and mins < -120:
            # Already left more than 2h ago — hide
            continue
        s["minutes_until_departure"] = mins
        items.append(_sanitize(s))

    items.sort(key=lambda x: x.get("departure_time", "99:99"))
    return {
        "user_id":         user_id,
        "schedules":       items,
        "count":           len(items),
        "current_time_my": now_my.strftime("%H:%M"),
        "today":           today_iso,
    }


@schedules_router.patch("/{schedule_id}")
def update_schedule(schedule_id: str, patch: ScheduleUpdate):
    from main import db
    ref = db.collection("user_schedules").document(schedule_id)
    if not ref.get().exists:
        raise HTTPException(404, "Schedule not found.")
    update_data = {k: v for k, v in patch.model_dump().items() if v is not None}
    update_data["updated_at"] = firestore.SERVER_TIMESTAMP
    ref.update(update_data)
    return {"status": "updated", "schedule_id": schedule_id}


@schedules_router.delete("/{schedule_id}")
def delete_schedule(schedule_id: str):
    from main import db
    ref = db.collection("user_schedules").document(schedule_id)
    if not ref.get().exists:
        raise HTTPException(404, "Schedule not found.")
    # Soft delete — preserves history for analytics & demo data
    ref.update({"active": False, "deleted_at": firestore.SERVER_TIMESTAMP})
    return {"status": "deleted", "schedule_id": schedule_id}


# ─────────────────────────────────────────────────────────────────────────────
# Proactive Agent — the core of the "Chat → Action" demo
# ─────────────────────────────────────────────────────────────────────────────

@schedules_router.post("/proactive-check")
def proactive_check(window_min: int = Query(default=60, ge=15, le=180),
                    user_id: Optional[str] = Query(default=None)):
    """
    Cloud Scheduler hits this every 15 minutes.

    For every active schedule whose departure_time is within `window_min`
    of now (in Malaysia time), the agent:
        1. Re-computes the best route given current traffic + weather.
        2. If the suggested mode differs from the user's preferred_mode
           OR a weather/congestion alert is worth flagging, writes a
           notification document. The frontend polls and shows it.

    Pass `user_id` to scope the run to a single user (used for demo
    & manual triggers from the UI).
    """
    from main import (db, get_osrm, get_traffic, build_options,
                      CO2, RM, call_gemini)

    now_my   = datetime.now(MYT)
    today    = now_my.strftime("%Y-%m-%d")
    weekday  = now_my.weekday()
    nowstamp = now_my.strftime("%H:%M")

    q = db.collection("user_schedules").where("active", "==", True)
    if user_id:
        q = q.where("user_id", "==", user_id)
    docs = list(q.stream())

    triggered: List[Dict[str, Any]] = []

    for doc in docs:
        s = doc.to_dict()
        if not _fires_today(s, today, weekday):
            continue

        mins_to_go = _minutes_until(s["departure_time"], now_my)
        if mins_to_go is None or not (15 <= mins_to_go <= window_min):
            # Outside watch window
            continue

        # Avoid double-firing — check whether we've already notified
        # for this schedule today.
        notif_id = f"{s['schedule_id']}_{today}"
        if db.collection("user_notifications").document(notif_id).get().exists:
            continue

        # ─── Run the route analysis ──────────────────────────────────────
        try:
            dist_km, base_t = get_osrm(
                s["start_lon"], s["start_lat"], s["end_lon"], s["end_lat"]
            )
            traffic, congestion = get_traffic(s["departure_time"])
            options = build_options(
                dist_km, base_t, traffic, congestion,
                s["departure_time"], has_vehicle=True,
            )
            weather = _get_weather(s["start_lat"], s["start_lon"])
        except Exception as e:
            log.warning(f"proactive_check skip {s['schedule_id']}: {e}")
            continue

        if not options:
            continue

        # Score by eco_score (carbon * 50% + cost * 25% + congestion * 25%)
        drive_co2  = dist_km * CO2["drive"] or 0.001
        drive_cost = dist_km * RM["petrol_per_km"] + RM["parking_city"] or 0.001
        for o in options:
            carbon_pct = 1 - (o["carbon_kg"] / drive_co2)
            cost_pct   = 1 - (o["cost_rm"]   / drive_cost)
            cong_map   = {"None": 1.0, "Very Low": 0.9, "Low": 0.75,
                          "Medium": 0.5, "High": 0.3, "Very High": 0.1}
            cong       = cong_map.get(o["congestion"], 0.5)
            o["eco_score"] = max(0, min(100, round(
                (carbon_pct * 0.50 + cost_pct * 0.25 + cong * 0.25) * 100
            )))
        options.sort(key=lambda x: -x["eco_score"])
        best = options[0]

        # Decide: is this a worth-pinging situation?
        preferred = s.get("preferred_mode")
        reasons: List[str] = []

        if weather and weather.get("rain_mm", 0) >= 1.0:
            reasons.append(
                f"rain expected ({weather['rain_mm']:.1f}mm in next hour)"
            )
        if congestion in ("High", "Very High"):
            reasons.append(f"{congestion.lower()} congestion on the road")
        if preferred and preferred.lower() != best["mode"].lower():
            reasons.append(
                f"a greener option saves "
                f"{round(max(0, drive_co2 - best['carbon_kg']), 2)} kg CO₂"
            )

        if not reasons:
            # Still send a friendly "you're good to go" ping if departure
            # is within 30 minutes — but mark it as low-priority.
            if mins_to_go > 30:
                continue

        # ─── Compose the message via Gemini ──────────────────────────────
        prompt = f"""You are EcoFlow's proactive commute agent.
A user has a saved schedule: "{s['label']}" — leaving at {s['departure_time']}
({mins_to_go} minutes from now). Trip is {round(dist_km, 1)} km.

Current conditions:
• Congestion: {congestion}
• Weather: {weather.get('summary', 'clear') if weather else 'unknown'}
• Best eco option: {best['mode']} — {best['time_mins']:.0f} min,
  RM {best['cost_rm']:.2f}, {best['carbon_kg']:.2f} kg CO₂, eco-score {best['eco_score']}/100.

User's usual choice: {preferred or 'unknown'}.

Why we're pinging them: {'; '.join(reasons)}.

Write ONE friendly notification (≤ 35 words, 1 emoji max) telling them:
1) what's happening on the road right now,
2) your specific recommendation including the suggested departure time,
3) the concrete benefit (minutes saved or kg CO₂ saved). Plain English."""

        fallback = (
            f"Heads up — leaving in {mins_to_go} min for {s['label']}. "
            f"With {congestion.lower()} traffic right now, "
            f"{best['mode']} would take ~{best['time_mins']:.0f} min "
            f"and emit only {best['carbon_kg']:.2f} kg CO₂. "
        )
        message = call_gemini(prompt, fallback)

        # ─── Write the notification doc ──────────────────────────────────
        notif = {
            "notif_id":         notif_id,
            "user_id":          s["user_id"],
            "schedule_id":      s["schedule_id"],
            "schedule_label":   s["label"],
            "departure_time":   s["departure_time"],
            "minutes_until":    mins_to_go,
            "message":          message,
            "recommended_mode": best["mode"],
            "recommended_time_mins": best["time_mins"],
            "recommended_cost_rm":   best["cost_rm"],
            "recommended_carbon_kg": best["carbon_kg"],
            "preferred_mode":   preferred,
            "reasons":          reasons,
            "weather":          weather,
            "congestion":       congestion,
            "distance_km":      round(dist_km, 2),
            "created_at":       firestore.SERVER_TIMESTAMP,
            "fired_at_my":      nowstamp,
            "read":             False,
            "dismissed":        False,
        }
        db.collection("user_notifications").document(notif_id).set(notif)
        triggered.append({
            "user_id":      s["user_id"][:6] + "***",
            "schedule_id":  s["schedule_id"],
            "label":        s["label"],
            "minutes_until": mins_to_go,
            "message":      message,
        })
        log.info(
            f"🔔 Proactive ping → {s['user_id'][:6]}*** for '{s['label']}' "
            f"({mins_to_go} min ahead): {message[:60]}…"
        )

    return {
        "checked_at_my":   nowstamp,
        "schedules_seen":  len(docs),
        "notifications":   len(triggered),
        "triggered":       triggered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Notifications — the UI polls these
# ─────────────────────────────────────────────────────────────────────────────

@schedules_router.get("/notifications/{user_id}")
def list_notifications(user_id: str, unread_only: bool = True, limit: int = 20):
    """Return recent proactive-agent notifications for a user."""
    from main import db

    q = db.collection("user_notifications").where("user_id", "==", user_id)
    if unread_only:
        q = q.where("read", "==", False)
    docs = list(q.limit(limit).stream())
    items = [_sanitize(d.to_dict()) for d in docs]
    # Most recent first
    items.sort(key=lambda x: x.get("created_at_iso") or x.get("notif_id", ""),
               reverse=True)
    return {"user_id": user_id, "notifications": items, "count": len(items)}


@schedules_router.post("/notifications/{notif_id}/read")
def mark_notification_read(notif_id: str):
    from main import db
    ref = db.collection("user_notifications").document(notif_id)
    if not ref.get().exists:
        raise HTTPException(404, "Notification not found.")
    ref.update({"read": True, "read_at": firestore.SERVER_TIMESTAMP})
    return {"status": "read", "notif_id": notif_id}


@schedules_router.post("/notifications/{notif_id}/accept")
def accept_recommendation(notif_id: str):
    """User tapped 'Use this' on the proactive ping."""
    from main import db
    ref = db.collection("user_notifications").document(notif_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(404, "Notification not found.")
    ref.update({
        "read": True,
        "accepted": True,
        "accepted_at": firestore.SERVER_TIMESTAMP,
    })
    return {"status": "accepted", "notif_id": notif_id, "data": _sanitize(snap.to_dict())}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fires_today(schedule: Dict[str, Any], today_iso: str, weekday: int) -> bool:
    """Decide whether a schedule fires on the given (today_iso, weekday)."""
    repeat = schedule.get("repeat", "once")
    if repeat == "once":
        return schedule.get("one_off_date") == today_iso
    if repeat == "daily":
        return True
    if repeat == "weekdays":
        return weekday < 5  # Mon..Fri
    if repeat == "custom":
        return weekday in (schedule.get("repeat_days") or [])
    return False


def _minutes_until(hhmm: str, now_my: datetime) -> Optional[int]:
    """Minutes from `now_my` until today's HH:MM (Malaysia time). Negative = past."""
    try:
        hh, mm = hhmm.split(":")
        target = now_my.replace(hour=int(hh), minute=int(mm),
                                second=0, microsecond=0)
        delta = (target - now_my).total_seconds() / 60.0
        return int(delta)
    except Exception:
        return None


def _get_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Hit Open-Meteo (no key needed) for the next-hour rain forecast."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=precipitation,temperature_2m,weather_code"
            "&forecast_hours=2&timezone=Asia%2FKuala_Lumpur"
        )
        r = requests.get(url, timeout=4)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        precip = (h.get("precipitation") or [0])[0] or 0.0
        temp   = (h.get("temperature_2m") or [None])[0]
        code   = (h.get("weather_code") or [0])[0]
        summary = _wmo_label(code)
        return {
            "rain_mm":     round(precip, 2),
            "temp_c":      temp,
            "wmo_code":    code,
            "summary":     summary,
        }
    except Exception as e:
        log.warning(f"weather fetch failed: {e}")
        return None


def _wmo_label(code: int) -> str:
    """Human-readable WMO weather code for the next hour."""
    if code == 0:                    return "clear"
    if code in (1, 2, 3):           return "partly cloudy"
    if code in (45, 48):            return "foggy"
    if code in (51, 53, 55, 56, 57): return "drizzling"
    if code in (61, 63, 65, 80, 81, 82): return "raining"
    if code in (71, 73, 75, 77, 85, 86): return "snow (unusual!)"
    if code in (95, 96, 99):        return "thunderstorm"
    return "uncertain"


def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Firestore Timestamps / Sentinel values into JSON-safe types."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        # Firestore returns DatetimeWithNanoseconds — make it a string.
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif type(v).__name__ in ("Sentinel", "ServerTimestamp"):
            # Skip un-resolved server timestamps (they appear pre-commit)
            continue
        else:
            out[k] = v
    return out
