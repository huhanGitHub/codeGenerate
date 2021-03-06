# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 30  # Maximum sentence length to consider

Transferred_Model_Path = 'save/model/transferred_model.pkl'
NL_Model_Path = 'save/model/nl_model.pkl'
Path_eval='save/test_pairs.txt'
Path_compare ='save/compare.txt'

Clue_s_task_Data_path='data/clue_summarization_task_data/toydata.txt'
NL_s_task='data/NL_summarization_task_data/toydata.txt'
Other_task='data/toydata.txt'

Test_ratio=0.2

Hidden_size=128
Output_size=128
num_layer=2

Teacher_forcing_ratio = 0.5

IterTimes = 200000

printTimes=5000

r_api=0.5
r_name=0.5

dropout=0.2