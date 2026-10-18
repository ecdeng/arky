from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _app_timezone() -> ZoneInfo:
    return ZoneInfo(os.getenv("APP_TIMEZONE", "America/Los_Angeles"))


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_app_timezone())
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_bounds(event: dict[str, Any]) -> tuple[datetime, datetime, bool]:
    start_data = event.get("start", {})
    end_data = event.get("end", {})
    zone = _app_timezone()

    if start_data.get("dateTime"):
        start = _parse_datetime(start_data["dateTime"])
        end = _parse_datetime(end_data["dateTime"])
        return start, end, False

    start_date = date.fromisoformat(start_data["date"])
    end_date = date.fromisoformat(end_data["date"])
    return (
        datetime.combine(start_date, time.min, tzinfo=zone),
        datetime.combine(end_date, time.min, tzinfo=zone),
        True,
    )


def _event_record(event: dict[str, Any]) -> dict[str, Any]:
    start, end, all_day = _event_bounds(event)
    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(untitled)"),
        "start": start.astimezone(_app_timezone()).isoformat(),
        "end": end.astimezone(_app_timezone()).isoformat(),
        "all_day": all_day,
        "notes": event.get("description", ""),
        "location": event.get("location", ""),
        "link": event.get("htmlLink", ""),
    }


class GoogleCalendarStore:
    def __init__(self, credentials_path: str | Path, token_path: str | Path, calendar_id: str = "primary"):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.calendar_id = calendar_id
        self._service = None

    def _credentials(self) -> Credentials:
        if not self.credentials_path.exists():
            raise RuntimeError(
                f"Google credentials not found at {self.credentials_path}. "
                "Download a Desktop OAuth client JSON from Google Cloud and save it there."
            )

        credentials = None
        if self.token_path.exists():
            credentials = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                credentials = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(credentials.to_json())

        return credentials

    def _service_client(self):
        if self._service is None:
            self._service = build(
                "calendar",
                "v3",
                credentials=self._credentials(),
                cache_discovery=False,
            )
        return self._service

    def _list_raw(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        result = (
            self._service_client()
            .events()
            .list(
                calendarId=self.calendar_id,
                timeMin=_utc_text(start),
                timeMax=_utc_text(end),
                singleEvents=True,
                showDeleted=False,
                orderBy="startTime",
            )
            .execute()
        )
        return result.get("items", [])

    def _list_records(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for event in self._list_raw(start, end):
            event_start, event_end, _ = _event_bounds(event)
            if event_start < end and event_end > start:
                records.append(_event_record(event))
        return records

    def list_events(self, start_iso: str | None = None, end_iso: str | None = None) -> str:
        start = _parse_datetime(start_iso) if start_iso else datetime.now(_app_timezone())
        end = _parse_datetime(end_iso) if end_iso else start + timedelta(days=30)
        return json.dumps(
            {
                "timezone": str(_app_timezone()),
                "events": self._list_records(start, end),
            }
        )

    def create_event(
        self,
        title: str,
        start_iso: str,
        end_iso: str,
        notes: str = "",
        location: str = "",
    ) -> str:
        start = _parse_datetime(start_iso)
        end = _parse_datetime(end_iso)
        if end <= start:
            return json.dumps({"ok": False, "error": "The end time must be after the start time."})

        conflicts = self._list_records(start, end)
        if conflicts:
            return json.dumps({"ok": False, "conflict": conflicts[0]})

        zone = str(_app_timezone())
        body = {
            "summary": title.strip(),
            "description": notes.strip(),
            "location": location.strip(),
            "start": {"dateTime": start.isoformat(), "timeZone": zone},
            "end": {"dateTime": end.isoformat(), "timeZone": zone},
        }
        created = (
            self._service_client()
            .events()
            .insert(calendarId=self.calendar_id, body=body)
            .execute()
        )
        return json.dumps({"ok": True, "event": _event_record(created)})

    def cancel_event(self, event_id: str) -> str:
        try:
            self._service_client().events().delete(
                calendarId=self.calendar_id,
                eventId=event_id,
            ).execute()
        except HttpError as error:
            if getattr(error, "resp", None) is not None and error.resp.status == 404:
                return json.dumps({"ok": False, "error": f"No Google Calendar event found with id {event_id}."})
            raise
        return json.dumps({"ok": True, "cancelled_id": event_id})

    def find_open_times(
        self,
        day: str,
        duration_minutes: int = 30,
        day_start: str = "09:00",
        day_end: str = "17:00",
    ) -> str:
        requested_day = date.fromisoformat(day)
        if duration_minutes <= 0 or duration_minutes > 12 * 60:
            return json.dumps({"ok": False, "error": "duration_minutes must be between 1 and 720."})

        zone = _app_timezone()
        window_start = datetime.combine(
            requested_day, time.fromisoformat(day_start), tzinfo=zone
        )
        window_end = datetime.combine(requested_day, time.fromisoformat(day_end), tzinfo=zone)
        if window_end <= window_start:
            return json.dumps({"ok": False, "error": "day_end must be after day_start."})

        busy = [
            (_parse_datetime(event["start"]), _parse_datetime(event["end"]))
            for event in self._list_records(window_start, window_end)
        ]
        slot_length = timedelta(minutes=duration_minutes)
        step = timedelta(minutes=30)
        available: list[dict[str, str]] = []
        cursor = window_start
        while cursor + slot_length <= window_end:
            slot_end = cursor + slot_length
            if not any(busy_start < slot_end and busy_end > cursor for busy_start, busy_end in busy):
                available.append({"start": cursor.isoformat(), "end": slot_end.isoformat()})
            cursor += step

        return json.dumps(
            {
                "ok": True,
                "timezone": str(zone),
                "date": day,
                "duration_minutes": duration_minutes,
                "available": available,
            }
        )
