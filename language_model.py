import re
import sys
import random
import math
import collections
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from constants import *


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        if self.lm:
            return self.lm.evaluate_text(text, smooth=True)
        else:
            raise ValueError("Language model not set for this Spell_Checker instance")

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            self.model_dict = collections.defaultdict(int) # a dictionary of the form {ngram:count}, holding counts of all ngrams
            self.token_frequency = collections.defaultdict(int)
            self.total_token_count = 0
            # in the specified text.
            # NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even
            # encouraged to) use a better data structure. However, you are requested to support this format for two
            # reasons: (1) It is very straight forward and force you to understand the logic behind LM, and (2) It
            # serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.

        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            if self.chars:
                # split text into individual characters
                words = list(text)
            else:
                # split text into words
                words = text.split()

            # construct the token frequencies dictionary
            for word in words:
                self.token_frequency[word] += 1

            # calculate total_token_count
            self.total_token_count = sum(self.token_frequency.values())

            # Add start and end tokens to the list of words to ensure that every n-gram has a full context of n
            # tokens. For example, in a trigram model (n=3), the first two words do not have a full context of 3
            # words, so we add 2 start tokens to the beginning of the list. Similarly, we add 1 end token to ensure
            # that every n-gram has a full context of 3 words.
            words = ['<s>'] * (self.n - 1) + words + ['</s>']

            # construct ngrams and count occurrences
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n])
                self.model_dict[ngram] += 1

        def get_token_frequency(self):
            """Returns the dictionary class object
            """
            return self.token_frequency

        def get_total_token_count(self):
            """Returns the dictionary class object
            """
            return self.total_token_count

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            if context is None:
                # sample a context from the model distribution
                contexts = [ngram[:-1] for ngram in self.model_dict.keys()]
                context = random.choice(contexts)
            else:
                context = context.split() if not self.chars else list(context)

            # add start token to the context
            context = ['<s>'] * (self.n - len(context)) + list(context)
            context = tuple(context[-(self.n - 1):])  # convert context to a tuple

            # generate the output sequence
            output = list(context)
            while len(output) < n:
                # get the counts of all possible next words given the current context
                counts = collections.defaultdict(int)
                for ngram, count in self.model_dict.items():
                    if ngram[:-1] == context:
                        counts[ngram[-1]] += count

                if counts:
                    # choose the word with the highest count (MLE)
                    next_word = max(counts, key=counts.get)
                else:
                    # if no next words are available, stop generating text
                    break

                output.append(next_word)
                context = context[1:] + (next_word,)

            # return the generated text as a string
            return ' '.join(output)

        def _check_for_oov(self, words):
            """Checks if any word in the given list is an out-of-vocabulary (OOV) word.
                If we have oov words, we need to apply smoothing to the language model.

            Args:
                words (list[str]): List of words to check.

            Returns:
                bool. True if any word in the list is an OOV word, False otherwise.
            """
            return not all(token in self.token_frequency.keys() for token in words)

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            # Initialize log probability to 0 and smooth False
            log_prob = 0
            smooth = False

            # Split the text into individual words
            words = text.split()

            # Pad the beginning and end of the text with special start and end tokens
            words = ['<s>'] * (self.n - 1) + words + ['</s>']

            total_count = sum(self.model_dict.values())

            if self._check_for_oov(words):
                smooth = True

            # Iterate over each ngram in the text
            for i in range(self.n - 1, len(words)):
                ngram = tuple(words[i - self.n + 1:i + 1])

                # Check if the vocabulary size is small enough to require smoothing
                if smooth:
                    # Use Laplace smoothing to calculate the probability of the ngram
                    prob = self.smooth(ngram)
                else:
                    # Calculate the probability of the ngram without smoothing
                    prob = self.model_dict.get(ngram, 0) / total_count

                # Take the logarithm of the probability and add it to the log probability
                try:
                    log_prob += math.log(prob)
                except ValueError:
                    # Handle the case where the probability is 0 and the logarithm is undefined
                    log_prob += 0

            # Return the log probability
            return log_prob

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            # Get the count of the ngram in the model
            count = self.model_dict.get(ngram, 0)

            # Get the total count of all ngrams in the model
            total_count = sum(self.model_dict.values())
            # Calculate the smoothed probability using Laplace smoothing
            # Add 1 to the count of the ngram and add the size of the vocabulary
            prob = (count + 1) / (total_count + len(self.model_dict))

            return prob


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)

    # Tokenize text into individual words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Rejoin words into a normalized string
    normalized_text = ' '.join(words)

    return normalized_text


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Etay Lorberboym', 'id': '314977596', 'email': 'etaylor@post.bgu.ac.il'}

