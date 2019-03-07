# pip3 install -r requirements.txt

# rm -rf datasets/webnlg_tensor*

# python3 preprocess.py \
#     -train_src datasets/train.src \
#     -train_tgt datasets/train.tgt \
#     -valid_src datasets/dev.src \
#     -valid_tgt datasets/dev.tgt \
#     -save_data datasets/webnlg_tensor

rm -rf webnlg_model300/*

rm -rf log300
mkdir log300

python3 train.py -data datasets/webnlg_tensor -save_model webnlg_model300/webnlg-model \
   -gpu_ranks 0 \
   -enc_layers 2 \
   -dec_layers 2 \
   -rnn_size 300 \
   -src_word_vec_size 500 \
   -tgt_word_vec_size 500 \
   -rnn_type LSTM \
   -train_steps 30000 \
   -valid_steps 5000 \
   -log_file log300/log.txt

rm -rf predict300
mkdir predict300

python3 translate.py -model webnlg_model300/webnlg-model_step_30000.pt \
  -src datasets/test.src -output predict300/pred30000.txt -replace_unk -verbose


python3 ../rdf2text/relex_predictions.py \
  -pred predict300/pred30000.txt \
  -relex datasets/test.relex \
  -output predict300/pred30000.relex


import os
os.chdir("../rdf2text")

! ./multi-bleu.perl  ../CS240_Project/datasets/test.ref1 ../CS240_Project/datasets/test.ref2 ../CS240_Project/datasets/test.ref3 < ../CS240_Project/predict300/pred30000.relex