# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 5  # Maximum sentence length to consider

MODEL_PATH = 'save/model.pkl'
Path_eval='save/test_pairs.txt'
Path_compare ='save/compare.txt'
Data_path='data/data.txt'

Test_ratio=0.2

Hidden_size=128

Teacher_forcing_ratio = 0.5

IterTimes = 200000