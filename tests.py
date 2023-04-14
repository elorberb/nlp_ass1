import unittest
from language_model import *


class MyTestCase(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("HELLO"), "hello")
        self.assertEqual(normalize_text("  hello  "), "hello")
        self.assertEqual(normalize_text("hello    world"), "hello world")

    def test_build_model_word(self):
        text = "The quick brown fox jumps over the lazy dog"
        expected_dict = {('<s>', '<s>', 'The'): 1,
                         ('<s>', 'The', 'quick'): 1,
                         ('The', 'quick', 'brown'): 1,
                         ('quick', 'brown', 'fox'): 1,
                         ('brown', 'fox', 'jumps'): 1,
                         ('fox', 'jumps', 'over'): 1,
                         ('jumps', 'over', 'the'): 1,
                         ('over', 'the', 'lazy'): 1,
                         ('the', 'lazy', 'dog'): 1,
                         ('lazy', 'dog', '</s>'): 1}
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.build_model(text)
        self.assertEqual(lm.get_model_dictionary(), expected_dict)

    def test_build_model_chars(self):
        text = "abbcabbcaaa"
        expected_dict = {('<s>', '<s>', 'a'): 1,
                         ('<s>', 'a', 'b'): 1,
                         ('a', 'b', 'b'): 2,
                         ('b', 'b', 'c'): 2,
                         ('b', 'c', 'a'): 2,
                         ('c', 'a', 'b'): 1,
                         ('c', 'a', 'a'): 1,
                         ('a', 'a', 'a'): 1,
                         ('a', 'a', '</s>'): 1,
                         }
        sc = Spell_Checker()
        lm = sc.Language_Model(chars=True)
        lm.build_model(text)
        self.assertEqual(lm.get_model_dictionary(), expected_dict)


if __name__ == '__main__':
    unittest.main()
