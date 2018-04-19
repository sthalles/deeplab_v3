# DeepLab_V3 Image Semantic Segmentation Network

Implementation of the Semantic Segmentation DeepLab_V3 CNN as described at [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf).

For a complete documentation of this implementation, check out the [blog post](https://sthalles.github.io/deep_segmentation_network/).

## Dependencies

- Python 3.x
- Numpy
- Tensorflow 1.4.0

## Downloads

Download the model checkpoints and dataset.

- [checkpoints](https://www.dropbox.com/sh/s7sx69pqjhrk0s4/AACXWCRd9JJ0zvcvDES9G3sba?dl=0)
- Dataset
  * [Option 1](https://mega.nz/#F!LlFCSaBB!1L_EoepUwhrHw4lHv1HRaA)
  * [Option 2](http://www.mediafire.com/?wx7h526chc4ar)

## Training and Eval

Before training, create a folder named ```checkpoints/``` inside the ```resnet/``` directory. Download the pre-trained [resnet-50](https://arxiv.org/abs/1603.05027) or [resnet-101](https://arxiv.org/abs/1603.05027) model from here, and place the files inside ```checkpoints/```.

To train this model run:

```
python train.py --starting_learning_rate=0.00001 --batch_norm_decay=0.997 --gpu_id=0 --resnet_model=resnet_v2_50
```

Check out the *train.py* file for more input argument options. Each run produces a folder inside the *tboard_logs* directory (create it if not there).

To evaluate the model, run the *test.py* file passing to it the *model_id* parameter (the name of the folder created inside *tboard_logs* during training).

```
python test.py --model_id=16645
```

## Datasets

To create the dataset, first make sure you have the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and the [Semantic Boundaries Dataset and Benchmark](http://home.bharathh.info/pubs/codes/SBD/download.html) datasets downloaded.

After, head to ```dataset/``` and run the ```CreateTfRecord.ipynb``` notebook. The ```custom_train.txt``` file contains the name of the images selected for training. This file is designed to use the Pascal VOC 2012 set as a **TESTING** set. Therefore, it doesn't contain any images from the VOC 2012 val dataset. For more info, see the **Training** section of [Deeplab Image Semantic Segmentation Network](https://sthalles.github.io/deep_segmentation_network/).

Obs. You can skip that part and direct download the datasets used in this experiment - See the **Downloads** section

## Results

- Pixel accuracy: ~91%
- Mean Accuracy: ~82%
- Mean Intersection over Union (mIoU): ~74%
- Frequency weighed Intersection over Union: ~86.

![Results](https://github.com/sthalles/sthalles.github.io/blob/master/assets/deep_segmentation_network/results1.png)
