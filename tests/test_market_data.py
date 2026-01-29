import unittest
from datetime import datetime
from unittest.mock import patch

from tools import market_data


class _DummyResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestMarketData(unittest.TestCase):
    def test_stooq_symbol_lowercases(self) -> None:
        self.assertEqual(market_data._stooq_symbol("AAPL"), "aapl.us")

    @patch("tools.market_data.urllib.request.urlopen")
    def test_fetch_stooq_daily_parses_csv(self, mock_urlopen) -> None:
        csv_body = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-01-01,1,1,1,10,100\n"
            "2026-01-02,1,1,1,20,200\n"
            "2026-01-03,1,1,1,,300\n"
        ).encode("utf-8")
        mock_urlopen.return_value = _DummyResponse(csv_body)

        rows = market_data._fetch_stooq_daily("AAPL")

        self.assertEqual(
            rows,
            [
                {"date": "2026-01-01", "close": 10.0},
                {"date": "2026-01-02", "close": 20.0},
            ],
        )

    @patch("tools.market_data._fetch_stooq_daily")
    def test_fetch_daily_closes_uses_stooq(self, mock_fetch) -> None:
        mock_fetch.return_value = [{"date": "2026-01-01", "close": 99.0}]

        series = market_data.fetch_daily_closes(["AAPL"], provider="stooq")

        self.assertIn("AAPL", series)
        self.assertEqual(series["AAPL"].df, [{"date": "2026-01-01", "close": 99.0}])

    @patch("tools.market_data._fetch_stooq_daily", return_value=None)
    @patch("tools.market_data.datetime")
    def test_fetch_daily_closes_fallback_series(self, mock_datetime, _mock_fetch) -> None:
        mock_datetime.utcnow.return_value = datetime(2026, 1, 27)

        series = market_data.fetch_daily_closes(["AAPL"], lookback_days=3, provider="stooq")

        rows = series["AAPL"].df
        self.assertEqual(
            [row["date"] for row in rows],
            ["2026-01-24", "2026-01-25", "2026-01-26"],
        )
        self.assertTrue(all(row["close"] == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
