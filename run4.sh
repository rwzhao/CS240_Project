# pip3 install -r requirements.txt

rm -rf datasets/webnlg_tensor4*

python3 preprocess.py \
    -train_src datasets/train.src \
    -train_tgt datasets/train.tgt \
    -valid_src datasets/dev.src \
    -valid_tgt datasets/dev.tgt \
    -save_data datasets/webnlg_tensor4

rm -rf webnlg_model4/*

rm -rf log4
mkdir log4

python3 train.py -data datasets/webnlg_tensor4 -save_model webnlg_model4/webnlg-model \
   -gpu_ranks 0 \
   -enc_layers 4 \
   -dec_layers 4 \
   -rnn_size 500 \
   -src_word_vec_size 500 \
   -tgt_word_vec_size 500 \
   -rnn_type LSTM \
   -train_steps 30000 \
   -valid_steps 5000 \
   -log_file log4/log.txt

rm -rf predict4
mkdir predict4

python3 translate.py -model webnlg_model4/webnlg-model_step_30000.pt \
  -src datasets/test.src -output predict4/pred.txt -replace_unk -verbose


python3 ../rdf2text/relex_predictions.py \
  -pred predict4/pred.txt \
  -relex datasets/test.relex \
  -output predict4/pred.relex