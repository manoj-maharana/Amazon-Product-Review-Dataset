# Amazon-CDs and Vinyl Data-set
# Background
As online marketplaces have been popular during the past decades, the online sellers and merchants ask 
their purchasers to share their opinions about the products they have bought. As a result, millions of 
reviews are being generated daily which makes it difficult for a potential consumer to make a good 
decision on whether to buy the product. Analyzing this enormous amount of opinions is also hard and 
time consuming for product manufacturers.

# About Data

http://jmcauley.ucsd.edu/data/amazon/

User Reviews provide feedback Data to a product. Every Amazon product review is summarized by a 
numerical rating But the heart of the feedback is in the text itself, not the rating The rating associated 
with every review is an integer from one to five stars. Ratings serve as supervised, multi-class labels for 
classifier, Review texts themselves are the core predictor Exploration of Natural Language Processing.

# The Goal:
Goal is to generate a term statistics using with the help of various libraries like( NLTK, spacy, Matplotlib)
• Vocabulary size with word frequencies

• N-grams, Tri-grams

• POS collections

• Most Frequent Noun Phrases

• Most Frequent Verb phrases

• NERs With their frequencies and types.

• Create a visualization graph

# Data Wrangling – NLP Pre-processing
• The final Data frame for the model will be drawn from the review Text Column

• The Overall column will serve as the ground truth labels.
# Sample Review-Text Data

['year debut 99 recording still best recording exception film 
soundtrack original stage show ever made extraordinary sound remarkable 
compared cast album decca rca capitol time period columbia always state 
art sound wise show performance fresh lilting moving martin excels 
honey bun pinza equally well enchanted evening nearly mine male chorus 
great job nothing like damewhile second cast album released lp kiss 
kate -also columbia - first also geared rpm reproduction cut maximum 
side could hold nothing seems rushed indeed columbia industry one 
better released sp disc side industry standard either quot disc quot 
disc - thus allowing u musicodd note parenthesis twin soliloques read 
wonder feel lyric nellie sings quotwonder id feelquot - nowhere 
parenthesized second title orignal lyric although song presented nothin 
like dame - listen guy - enunciate quotnothingquot dozen timesthe ]

# Lemmatization
• Words are reduced to their root form to ensure word usage consistency

• If learning is differentiated from its base version learn, we lose relational
context between two documents that use either word

• Lemmatization is a technique that takes into account context similarity
according to part-of-speech anatomy

• Stemming is another common approach although stemming only performs
Truncation

WordNetLemmatizer from NLTK library is used:
# NLP Pre-processing – Accents
Each review is normalized from longform UTF-8 to ASCII encoding

This removes accents in characters so words like naïve will simply reinterpreted as naive.

## NLP Pre-processing – Punctuations
• The preprocessed reviews are further cleaned by dropping punctuations

• Only spaces and alphanumeric characters are kept by replacing all RegExpattern [^\w\s] 
matches with a whitespace:

NLP Pre-processing – Lowercasing

Every letter is converted to lowercase

## NLP Pre-processing – Stop Words
• Stop words consist of most commonly used words that include:
pronouns (us, she, their) 
articles (a, an, the) 
prepositions (under, from, off)

• Stop words are not helpful in distinguishing a document from another and are therefore 
dropped
NLP Pre-processing – Single Whitespaces
RegEx pattern [\s]+ is used to ensure no more than a single whitespace separates words in sentences
Tokenization

• The pre-processed reviews make up our corpora, which is simply the collection of all our 
documents 

• Each review is tokenized or transformed into an ordered list of words

# Tokenized sample review text:

['im', 'big', 'fan', 'brainwavz', 's1', 'actually', 'headphone', 'yet', 'disappoint', 'product', 's1', 
'main', 'set', 'active', 'use', 'e', 'g', 'workouts', 'run', 'etc', 'since', 'flat', 'cable', 'durable', 
'resistant', 'tangle', 's5', 'keep', 'good', 'feature', 's1', 'add', 'sound', 'quality', 'rich', 'well', 'define', 
'thats', 'say', 's1', 'sound', 'poor', 'quite', 'good', 'fact', 's5', 'well', 'high', 'well', 'define', 
'midrange', 'punch', 'bass', 'come', 'clearly', 'without', 'move', 'harsh', 'territory', 'volume', 'push', 
's1s', 'overall', 'sound', 'quality', 'please', 'build', 'quality', 'seem', 'solid', 'solid', 's1', 'good', 'love', 
'flat', 'cable', 'know', 'thats', 'something', 'appreciate', 'everyone', 'work', 'wonderfully', 
'although', 'brainwavz', 'headset', 'come']

# Phrase Modeling – Bigrams
• Phrases are neighboring words that appear to convey one meaning as though they are a single 
word, like smart TV 

• Gensim’s built-in phraser is used 

• Higher phraser threshold means the more often two words must appear adjacent to be 
grouped into a phrase

# Sample bigram phrases: 
['hdmi_dvi', 'lens_without', 'time_forget', 'like_return', '2_00', 'fast_run', 'make_convenient', 
'point_think', 'matter_fact', 'although_make', 'actually_see', 'sure_problem', 'course_good', 
'get_catch', 'take_find', 'include_product', 'problem_design', 'work_everything', 
'standard_camera', '1080p_120hz', 'make_give', 'set_ipad', 'control_cable', 'nikon_brand', 
'really_beat', 'game_also', 'tiny_size', 'tiny_camera', 'use_default', 'color_come', 'get_12', 
'plug_network', 'piece_technology', 'light_fit', 'button_click', '4kb_qd', 'wheel_click', 
'wish_purchase', 'hold_device', 'ipod_phone', 'might_break', 'work_need', 'big_small', 
'tell_would', 'lot_high', 'noise_ratio', 'less_200', 'star_seem', 'design_camera', 
'camera_function']

# Phrase Modeling – Trigrams
• Trigrams are generated by applying another gensim phraser on top of a bigram phraser 

• If sd and card appear together often enough per the set parameters, the phraser groups them 
together as sd_card 

• If sd_card appears adjacent to the token reader in enough instances, the phraser further links 
them together as sd_card_reader

# Sample trigram phrases:
[(('--', '--', '--'), 421),
(('rock', 'n', 'roll'), 301),
(('ive', 'ever', 'heard'), 230),
(('one', 'best', 'album'), 224),
(('best', 'song', 'album'), 203),
(('dont', 'get', 'wrong'), 163),
(('one', 'best', 'song'), 154),
(('first', 'time', 'heard'), 151),
(('favorite', 'song', 'album'), 144),
(('cant', 'go', 'wrong'), 120)]

Count-based Feature Engineering

• For a machine learning model to work with text input, documents must first be vectorized or 
converted into containers of numerical values 

• Bag of Words is the classical approach of getting token frequency per document

• Term Frequency-Inverse Document Frequency (TF-IDF) is another approach where continuous 
values are assigned to tokens 

• Words that appear frequently overall are weighted lower because they do not establish saliency 
in a document 

• Words that are unique to some but not all documents are weighted higher because they help 
distinguish the documents from the others

Sample TF-IDF model: 

Word: address Weight: 0.113 Word: around Weight: 0.060

Word: arrive Weight: 0.093 Word: back Weight: 0.051

Word: bad Weight: 0.068 Word: big Weight: 0.126 

Word: come Weight: 0.046 Word: contact Weight: 0.103 

Word: could Weight: 0.054 Word: day Weight: 0.061 

Word: earlier Weight: 0.141 Word: ease Weight: 0.220

Final Dataframe
• All unique tokens in the entire corpora are vectorized in 100 dimensions making up the 
word_vec_df vocabulary 
• The word_vec_df is sliced by the words that appear in a given review and the mean is taken 
along each of the 100 dimensions 
• This singularizes multiple word embeddings into one observation for each product review 
• Finally, the ground truth label is imposed from the original Amazon dataset's overall series to 
create the final model_df dataframe
Results:
# Closing Thoughts – Conclusion
• Various NLP techniques and concepts were explored in the study 

• Though word embedding was central to building the model, pre-processing steps were crucial 

• The model actually extracts and quantifies context and therefore the essence of a review by its 
words make up the final dataframe 

• The multi-class, discrete classifier approach makes our model reliant on the distinction of each 
star-rating – if a one-star review was misclassified as a five-star review, the model is agnostic to 
how far off 1 and 5 are 

• It is more concerned in asking, "What makes a 5-star review different from a 4-star review?" 
than "Is this review more approving than criticizing?"
Closing Thoughts – Limits and Recommendations 

• The model will not be able to handle words that it has not encountered during training, it will 
simply drop new, unrecognizable words 

• No way of handling misspelled words – spell-check feature will only add to model complexity 

• Misspelled words will be taken as they are during training 

• As is usually the case in NLP, sarcasm or text that is intended to be ironic is interpreted by what 
is literally in the text and not by its underlying contex
