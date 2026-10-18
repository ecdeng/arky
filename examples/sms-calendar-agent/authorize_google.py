from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from google_calendar import SCOPES


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
credentials_path = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", BASE_DIR / "credentials.json"))
token_path = Path(os.getenv("GOOGLE_TOKEN_PATH", BASE_DIR / "data" / "google-token.json"))


def main() -> None:
    if not credentials_path.exists():
        raise SystemExit(
            f"Missing {credentials_path}. Download a Desktop OAuth client JSON "
            "from Google Cloud and save it there."
        )

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if credentials and credentials.valid:
        print(f"Google Calendar is already authorized; token: {token_path}")
        return

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        credentials = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json())
    print(f"Google Calendar authorization saved to {token_path}")


if __name__ == "__main__":
    main()
