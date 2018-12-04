cd ../../

currpath=`pwd`
# train the model
#python3 matchzoo/main.py --phase train --model_file ${currpath}/examples/HotpotQA/config/drmm_hotpotqa.config


# predict with the model

python3 matchzoo/main.py --phase predict --model_file ${currpath}/examples/HotpotQA/config/drmm_hotpotqa.config
