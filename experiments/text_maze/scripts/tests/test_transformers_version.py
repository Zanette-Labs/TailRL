#!/usr/bin/env python3
"""Tests for the transformers version gate (scripts/check_transformers.py).

Runs standalone (`python scripts/tests/test_transformers_version.py`) or under
pytest. The range logic needs no dependencies; the one test that touches the
installed transformers skips if it is not importable.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_transformers as ct  # noqa: E402


class TestParse(unittest.TestCase):
    def test_parses_major_minor(self):
        self.assertEqual(ct.parse("4.56.1"), (4, 56))
        self.assertEqual(ct.parse("4.51"), (4, 51))

    def test_tolerates_suffixes_and_prefixes(self):
        # dev builds, release candidates and a leading v all appear in the wild
        self.assertEqual(ct.parse("4.56.1.dev0"), (4, 56))
        self.assertEqual(ct.parse("5.0.0rc1"), (5, 0))
        self.assertEqual(ct.parse("v4.52.3"), (4, 52))

    def test_rejects_garbage(self):
        for bad in ("", "not-a-version", None):
            with self.assertRaises(ValueError):
                ct.parse(bad)


class TestSupportedRange(unittest.TestCase):
    def test_accepts_the_versions_the_paper_ran_on(self):
        # 4.51 is the container floor; 4.56.1 is what the reported runs used.
        for v in ("4.51.0", "4.51", "4.52.4", "4.56.1", "4.99.0"):
            self.assertTrue(ct.supported(v), v)

    def test_rejects_too_old(self):
        # < 4.51 misreads the checkpoint config -- the silent-0%-goal failure.
        for v in ("4.50.3", "4.0.0", "3.5.1"):
            self.assertFalse(ct.supported(v), v)

    def test_rejects_5x(self):
        # 5.x changes the model/generation APIs this verl fork targets.
        for v in ("5.0.0", "5.7.0", "6.1.0"):
            self.assertFalse(ct.supported(v), v)

    def test_boundaries_are_where_they_claim_to_be(self):
        self.assertEqual((ct.MIN, ct.MAX), ((4, 51), (5, 0)))
        self.assertFalse(ct.supported("4.50.99"))
        self.assertTrue(ct.supported("4.51.0"))
        self.assertTrue(ct.supported("4.99.99"))
        self.assertFalse(ct.supported("5.0.0"))


class TestCheck(unittest.TestCase):
    def test_check_is_silent_when_supported(self):
        self.assertIsNone(ct.check("4.56.1"))

    def test_check_exits_and_says_which_way_it_is_wrong(self):
        with self.assertRaises(SystemExit) as cm:
            ct.check("4.50.0")
        self.assertIn("too old", str(cm.exception))
        self.assertIn(ct.SUPPORTED, str(cm.exception))

        with self.assertRaises(SystemExit) as cm:
            ct.check("5.7.0")
        self.assertIn("too new", str(cm.exception))
        # the message must be actionable, not just a complaint
        self.assertIn("pip install", str(cm.exception))

    def test_requirements_txt_agrees_with_the_code(self):
        """A drifting pin would let pip install a version the gate then rejects."""
        req = os.path.join(os.path.dirname(ct.__file__), os.pardir, "requirements.txt")
        with open(req) as fh:
            line = next(l for l in fh if l.startswith("transformers"))
        self.assertIn(">=%d.%d" % ct.MIN, line)
        self.assertIn("<%d" % ct.MAX[0], line)


class TestInstalledVersion(unittest.TestCase):
    def test_the_environment_we_are_running_in_is_supported(self):
        try:
            version = ct.installed_version()
        except ImportError:
            self.skipTest("transformers not installed")
        self.assertTrue(
            ct.supported(version),
            "installed transformers %s is outside %s" % (version, ct.SUPPORTED),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
