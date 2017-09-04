import warnings
import arpa
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    

    # TODO implement the recognizer
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []
    Xlengths = test_set.get_all_Xlengths()
    wordlist = test_set.wordlist

    # Iterate the testing data. 
    for i in range(len(wordlist)):
      score_dict = {}
      best_word = ""
      max_score = float('-inf')
      X, lengths = Xlengths[i]

      # Iterate all possible words in models, and calculate the Log Liklihood
      for key in models:
        try:
          model = models[key]
          logL = model.score(X, lengths)
          score_dict[key] = logL

          # store the best guess words
          if logL > max_score:
            max_score = logL
            best_word = key
        except:
          score_dict[key] = -1

      # add prob dictionary and the best guess of each word
      probabilities.append(score_dict)
      guesses.append(best_word)
    
    return probabilities, guesses
    #raise NotImplementedError


def recognize_SLM(models: dict, test_set: SinglesData):
    # recognizer with SLM
    SLMmodel = arpa.loadf("ukn.3.lm")[0]
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    Xlengths = test_set.get_all_Xlengths()
    wordlist = test_set.wordlist
    # Iterate the testing data. 
    for video in test_set.sentences_index:
      # Create a word sequence of each video
      word_sequence = [wordlist[index] for index in test_set.sentences_index[video]]
      # Enumerate words in a video
      for i, word_index in enumerate(test_set.sentences_index[video]):
        score_dict = {}
        best_word = ""
        max_score = float('-inf')
        X, lengths = Xlengths[word_index]
        prefix = ""
        """
        # for 2-gram and 3-gram
        if i > 1:
          prefix = word_sequence[i-2: i]
        elif i == 1:
          prefix = word_sequence[i - 1]
        """

        # for 2-gram
        if i > 0:
          prefix = word_sequence[i - 1]

        # Iterate all possible words in models, and calculate the Log Liklihood
        for key in models:
          try:
            liklihood = SLMmodel.log_p(prefix +" " + key)
          except:
            liklihood = 0

          try:
            model = models[key]
            logL = model.score(X, lengths) + SLMmodel.log_p(key)
            score_dict[key] = logL

            # store the best guess words
            if logL > max_score:
              max_score = logL
              best_word = key
          except:
            score_dict[key] = -1

        # add prob dictionary and the best guess of each word
        probabilities.append(score_dict)
        guesses.append(best_word)
    
    
    return probabilities, guesses