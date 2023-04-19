import unittest
from language_model import *


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # Open the file and read its contents
        with open('corpus/the_raven.txt', 'r', encoding='utf-8') as file:
            self.the_raven = file.read()

        with open('corpus/big.txt', 'r', encoding='utf-8') as file:
            self.big = file.read()

    def test_normalize_text(self):
        # Test lowercase
        input_text = "This is a Sample TeXT."
        expected_output = "sample text"
        self.assertEqual(normalize_text(input_text), expected_output)

        # Test remove non-alphabetic
        input_text = "This is a sample text with 123 numbers and !@# special characters."
        expected_output = "sample text number special character"
        self.assertEqual(normalize_text(input_text), expected_output)

        # Test remove stopwords
        input_text = "This is a sample text with some stopwords."
        expected_output = "sample text stopwords"
        self.assertEqual(normalize_text(input_text), expected_output)

        # Test lemmatization
        input_text = "This text contains some lemmatizable words."
        expected_output = "text contains lemmatizable word"
        self.assertEqual(normalize_text(input_text), expected_output)

        # Test empty input
        input_text = ""
        expected_output = ""
        self.assertEqual(normalize_text(input_text), expected_output)

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
        expected_token_frequency = {'The': 1, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1,
                                    'the': 1, 'lazy': 1, 'dog': 1}

        expected_total_token_count = 9
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.build_model(text)
        self.assertEqual(lm.get_model_dictionary(), expected_dict)
        self.assertEqual(lm.get_total_token_count(), expected_total_token_count)
        self.assertEqual(lm.get_token_frequency(), expected_token_frequency)

    def test_check_for_oov(self):
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.model_dict = {('the', 'cat', 'sat'): 2, ('cat', 'sat', 'on'): 1, ('sat', 'on', 'the'): 1}

        # Test with input that contains no OOV words
        words1 = ['the', 'cat', 'sat', 'on', 'the']
        self.assertFalse(lm._check_for_oov(words1))

        # Test with input that contains OOV words
        words2 = ['the', 'dog', 'ran', 'on', 'the', 'mat']
        self.assertTrue(lm._check_for_oov(words2))

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
        expected_token_frequency = {'a': 5, 'b': 4, 'c': 2}
        expected_total_token_count = len(text)

        sc = Spell_Checker()
        lm = sc.Language_Model(chars=True)
        lm.build_model(text)

        self.assertEqual(lm.get_model_dictionary(), expected_dict)
        self.assertEqual(lm.token_frequency, expected_token_frequency)
        self.assertEqual(lm.total_token_count, expected_total_token_count)

    def test_generate(self):
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.build_model(self.the_raven)
        print(lm.generate(n=100))

    def test_evaluate_text(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.build_model(text)

        # Test the evaluation of a text with known log probability
        text = 'The quick brown fox jumps over the lazy dog.'
        expected_log_prob = -21.456879988908563
        log_prob = lm.evaluate_text(text)
        self.assertAlmostEqual(log_prob, expected_log_prob, places=6)

        # Test the evaluation of a text with unknown words
        text = 'The fast brown cat jumps over the lazy dog.'
        expected_log_prob = -25.13566585240802
        log_prob = lm.evaluate_text(text)
        self.assertAlmostEqual(log_prob, expected_log_prob, places=6)

    def test_smooth(self):
        expected_dict = {
            ('the', 'quick', 'brown'): 1,
            ('quick', 'brown', 'fox'): 2,
            ('brown', 'fox', 'jumps'): 1,
            ('fox', 'jumps', 'over'): 1,
            ('jumps', 'over', 'the'): 1,
            ('over', 'the', 'lazy'): 2,
            ('the', 'lazy', 'dog'): 1
        }
        sc = Spell_Checker()
        lm = sc.Language_Model()
        lm.model_dict = expected_dict
        prob = lm.smooth(('the', 'quick', 'brown'))
        self.assertAlmostEqual(prob, 2/16, places=3)

        prob = lm.smooth(('quick', 'brown', 'fox'))
        self.assertAlmostEqual(prob, 3/16, places=3)


if __name__ == '__main__':
    unittest.main()
