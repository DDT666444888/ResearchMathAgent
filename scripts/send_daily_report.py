#!/usr/bin/env python3
"""Send a plain-text daily progress report by email via the local SMTP relay
(127.0.0.1:25 — the same path the arc-daily-report / kaggle-watch services use).

Usage: send_daily_report.py <subject> <body_file> [--to addr]
Pure text only: no HTML, no tables — per the user's request.
"""
from __future__ import annotations

import argparse
import smtplib
import socket
from email.message import EmailMessage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("body_file")
    ap.add_argument("--to", default="sjtuytc@gmail.com")
    args = ap.parse_args()

    body = open(args.body_file, encoding="utf-8").read()
    msg = EmailMessage()
    msg["Subject"] = args.subject
    msg["From"] = f"rma-daily@{socket.getfqdn()}"
    msg["To"] = args.to
    msg.set_content(body)
    with smtplib.SMTP("127.0.0.1", 25, timeout=60) as smtp:
        smtp.send_message(msg)
    print(f"emailed {args.to}: {args.subject} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
