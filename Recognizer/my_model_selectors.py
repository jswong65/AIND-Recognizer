import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_BIC = 2 ** 31

        
        # Number of parameters can be calculated with:
        # Num of features = d
        # Num of HMM states = n
        # p = n*(n-1) + (n-1) + 2*d*n = = n^2 + 2*d*n - 1

        # TODO implement model selection based on BIC scores
        # Interate through different numbers of states
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_states)
            d = len(self.X[0]) # Num of features
            p = n_states ** 2 + 2 * d * n_states - 1 # Num of parameters
            try:
                logL = model.score(self.X, self.lengths)
                # Calculate the BIC
                BIC = -2 * logL + p * np.log(sum(self.lengths))
                # Store the current best score and model.
                if BIC < best_BIC:
                    best_BIC = BIC
                    best_model = model
            except:
                pass

		
        return best_model
		# raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_DIC = float('-inf')
        # Interate through different numbers of states
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_states)

            try:
                log_x_i = model.score(self.X, self.lengths)
                log_x_all = 0
                # Calculate SUM(log(P(X(all but i))
                for key, val in self.hwords.items():
                    if key == self.this_word:
                        pass
                    X, lengths = val
                    log_x_all += model.score(X, lengths)
                # Calculate the DIC    
                DIC = log_x_i - float(log_x_all) / (len(self.hwords.keys()) - 1)
                # Store the current best score and model.
                if DIC > best_DIC:
                        best_DIC = DIC
                        best_model = model
            except:
                pass
        return best_model
        # TODO implement model selection based on DIC scores
        # raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_model = None
        best_score = float('-inf')
        # Interate through different numbers of states
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            scores = []    
            if len(self.sequences) > 1: 
                n_split = 2 if len(self.sequences) <= 2 else 3
                split_method = KFold(shuffle=True, n_splits = n_split)
                # Iterate through different subsets of training and testing data
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        train_x, train_length = combine_sequences(cv_train_idx, self.sequences)
                        test_x, test_length = combine_sequences(cv_train_idx, self.sequences)
                        self.X = train_x
                        self.lengths = train_length
                        model = self.base_model(n_states) 
                        score = model.score(test_x, test_length)
                        scores.append(score) 
                    except:
                        pass
            # Store the current best score and model.
            if len(scores) > 0:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model

        return best_model
        # TODO implement model selection using CV
       # raise NotImplementedError
