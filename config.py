# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 5  # Maximum sentence length to consider

Transferred_Model_Path = 'save/model/transferred_model.pkl'
NL_Model_Path = 'save/model/nl_model.pkl'
Path_eval='save/test_pairs.txt'
Path_compare ='save/compare.txt'
Data_path='data/test.txt'

Test_ratio=0.2

Hidden_size=128

Teacher_forcing_ratio = 0.5

IterTimes = 20000

r_api=0.5
r_name=0.5

dropout=0.2