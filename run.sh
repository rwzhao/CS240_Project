pip3 install -r requirements.txt

rm -rf datasets/webnlg_tensor*

python3 preprocess.py \
    -train_src datasets/train.src \
    -train_tgt datasets/train.tgt \
    -valid_src datasets/dev.src \
    -valid_tgt datasets/dev.tgt \
    -save_data datasets/webnlg_tensor

rm -rf webnlg_model/*

rm -rf log
mkdir log

python3 train.py -data datasets/webnlg_tensor -save_model webnlg_model/webnlg-model \
   -gpu_ranks 0 \
   -enc_layers 3 \
   -dec_layers 3 \
   -rnn_size 500 \
   -src_word_vec_size 500 \
   -tgt_word_vec_size 500 \
   -rnn_type LSTM \
   -train_steps 30000 \
   -valid_steps 5000 \
   -log_file log/log.txt

rm -rf predict
mkdir predict

python3 translate.py -model webnlg_model/webnlg-model_step_30000.pt \
  -src datasets/test.src -output predict/pred.txt -replace_unk -verbose


python3 ../rdf2text/relex_predictions.py \
  -pred predict/pred.txt \
  -relex datasets/test.relex \
  -output predict/pred.relex