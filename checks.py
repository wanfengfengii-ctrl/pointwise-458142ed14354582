"""Frozen ordinary business regression, intentionally not a probability oracle."""
import json
import unittest
from fractions import Fraction as F
from pathlib import Path
from random import Random
from allocator import Allocator
from monitor import Monitor
from experiment import replay, simulate

class Checks(unittest.TestCase):
    def test_allocator(self):
        a = Allocator()
        self.assertEqual(a.probability(), F(1, 2))
        a.observe("A", 1)
        self.assertEqual(a.probability(), F(1, 2))
        a.observe("B", 0)
        self.assertEqual(a.probability(), F(3, 4))
        a.observe("A", 0)
        a.observe("B", 1)
        self.assertEqual(a.probability(), F(1, 2))
        a.observe("B", 1)
        self.assertEqual(a.probability(), F(1, 4))

    def test_monitor(self):
        m = Monitor()
        for q, expected in ((F(1,2), F(3,2)), (F(1,4), F(3)),
                            (F(3,4), F(4)), (F(1,4), F(8))):
            self.assertEqual(m.observe(q, "A", 1), expected)
        self.assertTrue(m.stopped)
        with self.assertRaises(RuntimeError):
            m.observe(F(1,2), "A", 0)
        z = Monitor()
        self.assertEqual(z.observe(F(3,4), "B", 1), 0)
        self.assertEqual(z.observe(F(1,4), "A", 1), 0)
        self.assertFalse(z.stopped)

    def test_replay(self):
        data = json.loads(Path(__file__).with_name("samples.json").read_text())
        pulled = []
        def tape():
            for row in data["tape"]:
                pulled.append(row)
                yield row
        result = replay(tape(), data["horizon"])
        expected = data["expected"]
        self.assertEqual(len(result.trace), expected["steps"])
        self.assertEqual(len(pulled), expected["steps"])
        self.assertEqual(list(result.counts), expected["counts"])
        self.assertEqual(str(result.evidence), expected["evidence"])
        self.assertEqual(result.stopped, expected["stopped"])
        self.assertEqual([str(s.probability) for s in result.trace],
                         expected["probabilities"])

    def test_zero_and_random(self):
        self.assertEqual(replay([], 0).counts, (0,0,0,0))
        self.assertEqual(replay([], 0).evidence, 1)
        a = simulate(F(0), Random(19), 16)
        self.assertFalse(a.stopped)
        self.assertEqual(a.evidence, 1)
        self.assertEqual(sum(a.counts), 16)
        self.assertEqual(simulate(F(2,3), Random(31)),
                         simulate(F(2,3), Random(31)))
        with self.assertRaises(ValueError):
            replay([], 1)
        with self.assertRaises(ValueError):
            replay([("A", True)], 1)

if __name__ == "__main__":
    unittest.main()
