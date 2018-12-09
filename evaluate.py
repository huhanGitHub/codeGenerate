import torch
from clue_summarization_task import EncoderRNN,  device, AttnDecoderRNN, evaluate
from dataHandle import Dictionary
from config import MODEL_PATH, dropout

import pickle
from clue_summarization_task import Hidden_size
from config import Path_eval, Path_compare


def main():
    checkpoint = torch.load(MODEL_PATH)
    encoder1_input_size = 2
    encoder1_hidden_size = Hidden_size

    encoder2_input_size = 2
    encoder2_hidden_size = Hidden_size

    decoder1_output_size = 128
    decoder1_hidden_size = Hidden_size

    input_lang = ''
    output_lang = ''

    for key, v in checkpoint.items():
        if key == 'Encoder1_state_dict':
            encoder1_input_size = v['embedding.weight'].shape[0]
            encoder1_hidden_size = v['embedding.weight'].shape[1]
        if key == 'Encoder2_state_dict':
            encoder2_input_size = v['embedding.weight'].shape[0]
            encoder2_hidden_size = v['embedding.weight'].shape[1]
        if key == 'Decoder_state_dict':
            decoder1_output_size = v['embedding.weight'].shape[0]
            decoder1_hidden_size = v['embedding.weight'].shape[1]

    with open('save/input.dic', 'rb') as f:
        input_dic = pickle.load(f)
        # print(type(input))

    with open('save/output.dic', 'rb') as f:
        output_dic = pickle.load(f)

    encoder1 = EncoderRNN(encoder1_input_size, encoder1_hidden_size)
    encoder2 = EncoderRNN(encoder2_input_size, encoder2_hidden_size)
    attn_decoder1 = AttnDecoderRNN(decoder1_hidden_size, decoder1_output_size, dropout_p=dropout).to(device)

    encoder1.load_state_dict(checkpoint['Encoder1_state_dict'])
    encoder1.eval()
    encoder2.load_state_dict(checkpoint['Encoder2_state_dict'])
    encoder2.eval()
    attn_decoder1.load_state_dict(checkpoint['Decoder_state_dict'])
    attn_decoder1.eval()

    with open(Path_eval, 'r') as f:
        test_pairs = f.readlines()
        score = 0
        for pair in test_pairs:
            # print(pair)
            pair = pair.split('\',\'')
            if len(pair) < 3:
                break
            pair[0] = pair[0][1:].strip()
            pair[2] = pair[2][:len(pair[2]) - 1].strip()
            pair[1] = pair[1].strip()

            sentence1 = pair[0]
            sentence2 = pair[1]
            # print(sentence)
            reference = pair[2]
            # print(reference)
            output_words, attentions = evaluate(encoder1, encoder2, attn_decoder1, sentence1, sentence2, input_dic,
                                                output_dic)
            # print(output_words)

            translate = str(reference) + '|||||' + (' '.join(output_words))
            print(translate)
            with open(Path_compare, 'a') as f2:
                f2.write(translate + '\n')


if __name__=='__main__':
    main()

