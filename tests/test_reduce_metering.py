"""Independent regression checks for malformed provider metering."""
import tempfile
import unittest
from pathlib import Path
from rma.reduction.backend import Ledger

class MeteringTests(unittest.TestCase):
    def test_malformed_usage_retains_hold(self):
        for bad in [True, False, 0.5, "2", float("inf"), float("nan"), -1, None]:
            for field in ["input_tokens", "output_tokens"]:
                with self.subTest(value=bad, field=field), tempfile.TemporaryDirectory() as d:
                    ledger = Ledger(Path(d)/"usage.jsonl", 10)
                    rec = ledger.reserve("test", 100)
                    usage = {"input_tokens": 1, "output_tokens": 1, field: bad}
                    ledger.finish(rec["id"], status="completed", settle_usage=usage)
                    self.assertEqual(ledger.reserved(), rec["reserved_usd"])
                    self.assertFalse(ledger.entries()[0].get("settled", False))

    def test_integer_usage_releases_unused_hold(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d)/"usage.jsonl", 10)
            rec = ledger.reserve("test", 100)
            ledger.finish(rec["id"], status="completed",
                          settle_usage={"input_tokens": 1, "output_tokens": 1})
            self.assertAlmostEqual(ledger.reserved(), 0.0006)
            self.assertTrue(ledger.entries()[0]["settled"])
