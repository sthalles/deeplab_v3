# DeepLab_V3 Image Semantic Segmentation Network

Implementation of the Semantic Segmentation DeepLab_V3 CNN as described in: https://arxiv.org/pdf/1606.00915.pdf

For a complete documentation of this implementation, check out the [blog post](https://sthalles.github.io/deep_segmentation_network/).

## Dependencies

- Python 3.x
- Numpy

## Training and Eval

To train this model run:

```
python train.py --starting_learning_rate=0.00001 --batch_norm_decay=0.997 --gpu_id=0 --resnet_model=resnet_v2_50
```

Check out the *train.py* file for more input argument options. Each run produces a folder inside the *tboard_logs* directory (create it if not there).

To evaluate the model, run the *test.py* file passing to it the model_id parameter (the name of the folder created during training).

```
python test.py --model_id=16645
```

## Results

- Pixel accuracy: ~91%
- Mean Accuracy: ~82%
- Mean Intersection over Union (mIoU): ~74%
- Frequency weighed Intersection over Union: ~86.

![Results](https://github.com/sthalles/sthalles.github.io/blob/master/assets/deep_segmentation_network/results1.png)
