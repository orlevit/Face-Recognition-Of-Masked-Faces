echo '-------------------- nomask ------------------------------'
python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/datasets/alfw  --target nomask_lfw --model /home/orlev/work/project/insightface/models/model-r100-ii/model,0  --roc-name nomask_nomask --threshold 1.6| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/eyeMask  --target eyeMask_lfw  --model /home/orlev/work/project/insightface/models/model-r100-ii/model,0  --roc-name nomask_eyemask --threshold 1.6| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/hatMask  --target hatMask_lfw  --model /home/orlev/work/project/insightface/models/model-r100-ii/model,0 --roc-name nomask_hatmask --threshold 1.6| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/scarfMask  --target scarfMask_lfw  --model /home/orlev/work/project/insightface/models/model-r100-ii/model,0 --roc-name nomask_scarfmask --threshold 1.6| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/coronaMask  --target coronaMask_lfw  --model /home/orlev/work/project/insightface/models/model-r100-ii/model,0 --roc-name nomask_coronamask --threshold 1.6| tail -3 |sed -n 1,2p
echo '-------------------- eyemask ------------------------------'
python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/datasets/alfw  --target nomask_lfw --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_eye_masked/model,1  --roc-name eyemask_nomask --threshold 1.51| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/eyeMask  --target eyeMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_eye_masked/model,1  --roc-name eyemask_eyemask --threshold 1.51| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/hatMask  --target hatMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_eye_masked/model,1  --roc-name eyemask_hatmask --threshold 1.51| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/scarfMask  --target scarfMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_eye_masked/model,1  --roc-name eyemask_scarfmask --threshold 1.51| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/coronaMask  --target coronaMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_eye_masked/model,1  --roc-name eyemask_coronamask --threshold 1.51| tail -3 |sed -n 1,2p
echo '-------------------- hatmask ------------------------------'
python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/datasets/alfw  --target nomask_lfw --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_hat_masked/model,1  --roc-name hatmask_nomask --threshold 1.56| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/eyeMask  --target eyeMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_hat_masked/model,1  --roc-name hatmask_eyemask --threshold 1.56| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/hatMask  --target hatMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_hat_masked/model,1  --roc-name hatmask_hatmask --threshold 1.56| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/scarfMask  --target scarfMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_hat_masked/model,1  --roc-name hatmask_scarfmask --threshold 1.56| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/coronaMask  --target coronaMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_hat_masked/model,1  --roc-name hatmask_coronamask --threshold 1.56| tail -3 |sed -n 1,2p
echo '-------------------- scarfmask ------------------------------'
python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/datasets/alfw  --target nomask_lfw --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_scarf_masked/model,1  --roc-name scarfmask_nomask --threshold 1.62| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/eyeMask  --target eyeMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_scarf_masked/model,1  --roc-name scarfmask_eyemask --threshold 1.62| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/hatMask  --target hatMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_scarf_masked/model,1  --roc-name scarfmask_hatmask --threshold 1.62| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/scarfMask  --target scarfMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_scarf_masked/model,1  --roc-name scarfmask_scarfmask --threshold 1.62| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/coronaMask  --target coronaMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_scarf_masked/model,1  --roc-name scarfmask_coronamask --threshold 1.62| tail -3 |sed -n 1,2p
echo '-------------------- coronamask ------------------------------'
python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/datasets/alfw  --target nomask_lfw --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_corona_masked/model,1  --roc-name coronamask_nomask --threshold 1.55| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/eyeMask  --target eyeMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_corona_masked/model,1  --roc-name coronamask_eyemask --threshold 1.55| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/hatMask  --target hatMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_corona_masked/model,1  --roc-name coronamask_hatmask --threshold 1.55| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/scarfMask  --target scarfMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_corona_masked/model,1  --roc-name coronamask_scarfmask --threshold 1.55| tail -3 |sed -n 1,2p && python /home/orlev/work/project/insightface/recognition/common/test_model_new.py  --data-dir /home/orlev/work/project/Expression-old/Masks_pics/test_round2_7_lfw/test_round2_7_ver2/coronaMask  --target coronaMask_lfw  --model /home/orlev/work/project/scripts/run/models/transfer_learning/r100-arcface-ver2_corona_masked/model,1  --roc-name coronamask_coronamask --threshold 1.55| tail -3 |sed -n 1,2p
