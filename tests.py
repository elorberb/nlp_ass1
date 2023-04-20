import unittest
from language_model import *


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # Open the file and read its contents
        with open('corpus/the_raven.txt', 'r', encoding='utf-8') as file:
            self.the_raven = file.read()

        with open('corpus/big.txt', 'r', encoding='utf-8') as file:
            self.big = file.read()

        sc = Spell_Checker()
        self.lm = sc.Language_Model()
        self.lm_chars = sc.Language_Model(chars=True)

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
        self.lm.build_model(text)
        self.assertEqual(self.lm.get_model_dictionary(), expected_dict)
        self.assertEqual(self.lm.get_total_token_count(), expected_total_token_count)
        self.assertEqual(self.lm.get_token_frequency(), expected_token_frequency)

    def test_check_for_oov(self):
        text = "the cat sat on the mat"
        self.lm.build_model(text)

        # Test with input that contains no OOV words
        words1 = ['the', 'cat', 'sat', 'on', 'the']
        self.assertFalse(self.lm._check_for_oov(words1))

        # Test with input that contains OOV words
        words2 = ['the', 'dog', 'ran', 'on', 'the', 'mat']
        self.assertTrue(self.lm._check_for_oov(words2))

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
        self.lm_chars.build_model(text)

        self.assertEqual(self.lm_chars.get_model_dictionary(), expected_dict)
        self.assertEqual(self.lm_chars.token_frequency, expected_token_frequency)
        self.assertEqual(self.lm_chars.total_token_count, expected_total_token_count)

    def test_generate(self):
        self.lm.build_model(self.the_raven)
        print(self.lm.generate(n=100))

    def test_generate_chars(self):
        self.lm_chars.build_model(self.the_raven)
        print(self.lm_chars.generate(n=100))

    def test_evaluate_text_no_smoothing(self):
        # Prepare a model_dict and total_token_count for this test
        self.lm.model_dict = {('<s>', '<s>', 'hello'): 1, ('<s>', 'hello', 'world'): 1, ('hello', 'world', '</s>'): 1}
        self.lm.total_token_count = 3

        text = "hello world"
        expected_log_prob = math.log(1 / 3) * 3
        actual_log_prob = self.lm.evaluate_text(text)

        self.assertAlmostEqual(expected_log_prob, actual_log_prob)

    def test_evaluate_text_with_smoothing(self):
        # Prepare a model_dict, total_token_count, and vocabulary for this test
        self.lm.model_dict = {('<s>', '<s>', 'hello'): 1, ('<s>', 'hello', 'world'): 1, ('hello', 'world', '</s>'): 1}
        self.lm.total_token_count = 2
        self.lm.token_frequency = {'hello': 1, 'world': 1}

        text = "hello unknown"  # 'unknown' is an out-of-vocabulary word
        expected_log_prob = math.log(self.lm.smooth(('<s>', '<s>', 'hello'))) + math.log(
            self.lm.smooth(('<s>', 'hello', 'unknown'))) + math.log(self.lm.smooth(('hello', 'unknown', '</s>')))
        actual_log_prob = self.lm.evaluate_text(text)

        self.assertAlmostEqual(expected_log_prob, actual_log_prob)

    def test_evaluate_text_with_zero_probability(self):
        # Prepare a model_dict and total_token_count for this test
        self.lm.model_dict = {('<s>', '<s>', 'hello'): 1, ('<s>', 'hello', 'world'): 1, ('hello', 'world', '</s>'): 1}
        self.lm.total_token_count = 3
        self.lm.token_frequency = {'hello': 1, 'world': 1}

        text = "world hello"  # 'unknown' is an out-of-vocabulary word
        expected_log_prob = 0  # Since there is a zero probability, the log probability will be 0
        actual_log_prob = self.lm.evaluate_text(text)

        self.assertEqual(expected_log_prob, actual_log_prob)

    def test_evaluate_text_character_based(self):
        # Prepare a model_dict and total_token_count for this test
        self.lm_chars.model_dict = {('<s>', '<s>', 'h'): 1, ('<s>', 'h', 'e'): 1, ('h', 'e', 'l'): 1,
                                    ('e', 'l', 'l'): 1, ('l', 'l', 'o'): 1, ('l', 'o', '</s>'): 1}
        self.lm_chars.total_token_count = 5
        self.lm_chars.token_frequency = {'h': 1, 'e': 1, 'l': 2, 'o': 1}

        text = "hello"
        expected_log_prob = math.log(1 / 5) * 6
        actual_log_prob = self.lm_chars.evaluate_text(text)

        self.assertAlmostEqual(expected_log_prob, actual_log_prob)

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

        self.lm.model_dict = expected_dict
        self.lm.total_token_count = 9
        prob = self.lm.smooth(('the', 'quick', 'brown'))
        self.assertAlmostEqual(prob, 2 / 16, places=3)

        prob = self.lm.smooth(('quick', 'brown', 'fox'))
        self.assertAlmostEqual(prob, 3 / 16, places=3)


if __name__ == '__main__':
    unittest.main()
