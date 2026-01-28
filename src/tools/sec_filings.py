from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MAX_CHARS = 200_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    try:
        return str(value)
    except Exception:
        return None


def _call_maybe(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _get_value(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return _call_maybe(getattr(obj, name))
    return None


def _to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        return str(value)
    except Exception:
        return None


def _extract_from_document(doc: Any) -> Optional[str]:
    if doc is None:
        return None
    for attr in ("text", "content", "html", "body", "raw", "document"):
        val = _call_maybe(getattr(doc, attr, None))
        if val:
            txt = _to_text(val)
            if txt:
                return txt
    return None


def _extract_filing_text(filing: Any) -> Tuple[Optional[str], str]:
    if filing is None:
        return None, "none"

    for attr in ("text", "document", "primary_document", "html", "xml"):
        val = _call_maybe(getattr(filing, attr, None))
        if val:
            if isinstance(val, (str, bytes)):
                txt = _to_text(val)
                if txt:
                    return txt, f"filing.{attr}"
            else:
                txt = _extract_from_document(val)
                if txt:
                    return txt, f"filing.{attr}"

    attachments = getattr(filing, "attachments", None)
    if attachments:
        primary = _get_value(attachments, ["primary_document", "primary", "main_document"])
        if primary:
            txt = _extract_from_document(primary)
            if txt:
                return txt, "attachments.primary"
        try:
            for att in attachments:
                txt = _extract_from_document(att)
                if txt:
                    return txt, "attachments.iter"
        except Exception:
            pass

    return _to_text(str(filing)), "stringify"


def _trim_text(text: Optional[str], max_chars: int) -> Tuple[Optional[str], bool, int]:
    if text is None:
        return None, False, 0
    original_len = len(text)
    if max_chars <= 0 or original_len <= max_chars:
        return text, False, original_len
    return text[:max_chars], True, original_len


def init_sec_identity(require: bool = True) -> Optional[str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua:
        from edgar import set_identity

        set_identity(ua)
        return ua
    if require:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Please set it to a descriptive value like "
            "'Your Name <you@example.com>' to access SEC data."
        )
    return None


def _list_filings(filings: Any, limit: int) -> List[Any]:
    if limit <= 0:
        limit = 1

    if hasattr(filings, "latest"):
        try:
            if limit == 1:
                latest = filings.latest()
                return [latest] if latest is not None else []
            try:
                latest_n = filings.latest(limit)
                if latest_n is not None:
                    return list(latest_n)
            except TypeError:
                pass
        except Exception:
            pass

    if hasattr(filings, "head"):
        try:
            head = filings.head(limit)
            return list(head) if not isinstance(head, list) else head
        except Exception:
            pass

    try:
        return list(filings)[:limit]
    except Exception:
        pass

    try:
        return filings.to_list()[:limit]
    except Exception:
        return []


def _filing_to_dict(
    filing: Any,
    include_text: bool,
    max_chars: int,
    include_attachments: bool,
) -> Dict[str, Any]:
    form = _get_value(filing, ["form", "form_type"])
    filing_date = _to_iso_date(_get_value(filing, ["filing_date", "date_filed", "filing_date_str"]))
    report_date = _to_iso_date(_get_value(filing, ["report_date", "period_of_report"]))
    accession_no = _get_value(filing, ["accession_no", "accession_number", "accession"])
    cik = _get_value(filing, ["cik", "company_cik"])
    company_name = _get_value(filing, ["company", "company_name", "issuer_name", "name"])
    file_number = _get_value(filing, ["file_number", "file_no"])
    primary_document = _get_value(
        filing, ["primary_document", "document", "primary_doc", "document_name"]
    )
    url = _get_value(filing, ["url", "link", "href"])

    payload: Dict[str, Any] = {
        "form": _json_safe(form),
        "filing_date": filing_date,
        "report_date": report_date,
        "accession_no": _json_safe(accession_no),
        "cik": _json_safe(cik),
        "company_name": _json_safe(company_name),
        "file_number": _json_safe(file_number),
        "primary_document": _json_safe(primary_document),
        "url": _json_safe(url),
    }

    if include_text:
        raw_text, source = _extract_filing_text(filing)
        text, truncated, original_len = _trim_text(raw_text, max_chars)
        payload.update(
            {
                "text": text,
                "text_source": source,
                "text_chars": original_len,
                "text_truncated": truncated,
            }
        )

    if include_attachments:
        attachments = getattr(filing, "attachments", None)
        if attachments is not None:
            payload["attachments"] = _attachments_to_list(attachments)

    return payload


def _attachments_to_list(attachments: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if attachments is None:
        return out

    def _as_dict(att: Any) -> Dict[str, Any]:
        return {
            "sequence": _json_safe(_get_value(att, ["sequence", "seq"])),
            "filename": _json_safe(_get_value(att, ["filename", "file_name", "document", "name"])),
            "description": _json_safe(_get_value(att, ["description", "descr"])),
            "type": _json_safe(_get_value(att, ["type", "document_type", "doctype"])),
            "url": _json_safe(_get_value(att, ["url", "link", "href"])),
            "is_primary": bool(_get_value(att, ["primary", "is_primary"])),
        }

    try:
        for att in attachments:
            out.append(_as_dict(att))
        return out
    except Exception:
        pass

    try:
        out.append(_as_dict(attachments))
    except Exception:
        pass
    return out


def build_sec_filings_payload(
    ticker: str,
    form: str = "10-Q",
    limit: int = 1,
    include_text: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_attachments: bool = False,
    require_user_agent: bool = True,
) -> Dict[str, Any]:
    if not ticker:
        raise ValueError("Ticker is required")

    ua = init_sec_identity(require=require_user_agent)

    from edgar import Company

    company = Company(ticker)
    filings = company.get_filings(form=form)
    items = _list_filings(filings, limit)

    payload = {
        "ticker": ticker.upper(),
        "form": form.upper(),
        "retrieved_at": _utc_now_iso(),
        "user_agent": ua,
        "count": len(items),
        "filings": [
            _filing_to_dict(
                filing,
                include_text=include_text,
                max_chars=max_chars,
                include_attachments=include_attachments,
            )
            for filing in items
        ],
    }
    return payload


def _default_output_path(ticker: str, form: str) -> Path:
    safe_form = "".join(ch if ch.isalnum() else "_" for ch in form.upper())
    return DEFAULT_OUTPUT_DIR / f"sec_filings_{ticker.upper()}_{safe_form}.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch SEC filings and save to JSON.")
    p.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    p.add_argument("--form", default="10-Q", help="Filing form type, e.g. 10-K, 10-Q, 8-K")
    p.add_argument("--limit", type=int, default=1, help="Number of latest filings to fetch")
    p.add_argument("--out", default="", help="Output JSON path (default: outputs/sec_filings_TICKER_FORM.json)")
    p.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file")
    p.add_argument("--no-text", action="store_true", help="Skip filing text to keep output small")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max characters for filing text")
    p.add_argument(
        "--include-attachments",
        action="store_true",
        help="Include attachment metadata (no attachment content)",
    )
    p.add_argument(
        "--allow-missing-ua",
        action="store_true",
        help="Allow running without SEC_USER_AGENT (not recommended)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_sec_filings_payload(
        ticker=args.ticker,
        form=args.form,
        limit=args.limit,
        include_text=not args.no_text,
        max_chars=args.max_chars,
        include_attachments=args.include_attachments,
        require_user_agent=not args.allow_missing_ua,
    )

    if args.stdout:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    out_path = Path(args.out) if args.out else _default_output_path(args.ticker, args.form)
    write_json(out_path, payload)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
