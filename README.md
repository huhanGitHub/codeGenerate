# codeGenerate
## (1) How to run the mdoel
run *clue_summarization_task.py* firstly, and the script will generate a transferred encoder, and after running the script, run *NL_summarization_task.py*.

## (2) Dataset
Our dataset is collect form three open source dataset:BigCloneBench(https://github.com/clonebench/BigCloneBench/blob/master/README.md), 
concode(https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W), Awesome Java(https://github.com/akullpp/awesome-java) and JDK1.8.1 141 source code. Our dataset is on (https://drive.google.com/open?id=1nOuZjSS9lUqWfQptUOhfX9kNKd_FeCkn).
We prepare a toy-data for debugging, it's in the 'data' directory.

## (3) Other models
Run *API_plus_name.py* for Seq2Seq with clues as input only model;
run *NL_only.py for Seq2Seq* with NL as input only model;
run *without_transferred.py* for Seq2Seq without transferred clues model.

## (4) updating
