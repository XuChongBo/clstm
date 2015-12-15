######## generate train data
../tools/tiku_ocr_char_gen -o ~/data/tt -t sym_out_map.txt

######## recoginze batchly
load="./my-500-199000.clstm" ../clstmocr train.txt

######## do train



############
python flat_dir_same_name_gif.py  ~/data/tiku_images/gif/chinese ~/data/tiku_images/png/chinese

######
generate the samples


#########
