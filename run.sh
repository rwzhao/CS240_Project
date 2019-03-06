pip3 install -r requirements.txt

rm -rf datasets/webnlg_tensor*

python3 preprocess.py \
    -train_src datasets/train.src \
    -train_tgt datasets/train.tgt \
    -valid_src datasets/dev.src \
    -valid_tgt datasets/dev.tgt \
    -save_data datasets/webnlg_tensor

rm -rf webnlg_model*

python3 train.py -data datasets/webnlg_tensor -save_model webnlg_model/webnlg-model \
   -enc_layers 1
   -dec_layers 1
   -rnn_size 500
   -src_word_vec_size 500 \
   -tgt_word_vec_size 500 \
   -rnn_type LSTM \
   -train_steps 30000 \
   -valid_steps 5000 \
   -log_file log.txt

# python3 translate.py -model webnlg_model/webnlg-model_acc_XX.XX_ppl_XXX.XX_eX.pt \
#   -src datasets/test.src -output pred.txt -replace_unk -verbose