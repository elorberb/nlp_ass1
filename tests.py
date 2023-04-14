import unittest
from language_model import *


class MyTestCase(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("HELLO"), "hello")
        self.assertEqual(normalize_text("  hello  "), "hello")
        self.assertEqual(normalize_text("hello    world"), "hello world")


if __name__ == '__main__':
    unittest.main()
