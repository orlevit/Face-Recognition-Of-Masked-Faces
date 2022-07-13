#!/bin/bash

# run `python ijb.py --help` for more information
python -u ijb.py \
--model-prefix /home/face/insightface/recognition/partial_fc/mxnet/evaluation/glint360k_r100FC_0.1/model \
--image-path /data/anxiang/IJB_release/IJBC \
--result-dir ./results/test \
--model-epoch 0 \
--gpu 0,1,2,3 \
--target IJBC \
--job partial_fc \
--batch-size 256 \
-es 512
python -u ijb_combinedV1.py --model-prefix /RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning --image-path /RG/rg-tal/orlev/downloads/aaa  --result-dir ./results/test --model-epoch 0 --gpu 0 --target IJBC --job partial_fc --batch-size 256 -es 512
