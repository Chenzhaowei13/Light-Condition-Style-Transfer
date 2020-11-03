python3 -u test_erfnet.py CULane ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 0 1 \
                          --resume trained/_erfnet_model_best.pth.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 10 \
                          -b 5
