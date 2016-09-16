Deep Group Happiness
===============================

We release Deep Group Happiness framework, presented the paper: 

A. Cerekovic: A deep look into group happiness prediction from images, in Proceedings of the 2016 ACM on International Conference on Multimodal Interaction ICMI'16, preprint [2]. 

The framework uses Tensorflow models to predict group happiness from images, as follows:

1. Face Detection model, trained with [Tensorbox] (https://github.com/Russell91/TensorBox) [4] 
2. Face Happiness prediction model
3. Face-CENTRIST model (an LSTM-based model) which predicts group happiness based on spatial distribution of faces in the image, their facial expression of happiness, and CENTRIST image descriptor. The model is trained on the HAPPEI training set.

Details about framework are given in [2]. The only modification to the work from [2] is the Face Happiness prediction model. Hereby we use VGG16, as in later experiments VGG16 slightly outperformed GoogLeNet-FC for the task of individual face happiness prediction. Consequently, new Face-CENTRIST model is trained with data extracted from the VGG16 model. The version at this moment has RMSE 0.54 over the HAPPEI validation set, provided in [1].

All models are released under the [Creative Commons Attribution-NonCommercial License](https://creativecommons.org/licenses/by-nc/4.0/) [5] and are free to use for non-commertial purposes. 

If you find the framework/models to be useful in your research work, a citation to the following paper would be appreciated:

[2]: A. Cerekovic: A deep look into group happiness prediction from images, in Proceedings of the 2016 ACM on International Conference on Multimodal Interaction ICMI'16, preprint

Happy coding (and research)!


Downloading models 
------------------
To set up the environment, run bash script download_data.sh. The script will download trained models and prerequisites for model training and testing.

Demo
----

Upon running the download_data.sh, the models will be located in the ./data/models/ directory. 

Due to the EULA copyright, we are unable to provide the original HAPPEI dataset. Demo is done on images obtained from the Flicker and some of the images from author's private collection. Just run demo.py and observe the result.

Training and evaluation
-----------------------
 
If one wants to retrain the Face-CENTRIST model, we provide training and evaluation scripts. 

Script train.py serves to train the model. The script has to be called with the following arguments:

```
--data_dir="./data/fake_HAPPEI"
```

Where data_dir points to the training and validation sets. The structure of the data_dir directory has to be as follows:

```
--data_dir
    -- \images
    -- \CENTRIST
    -- data_dir_training.csv
    -- data_dir_validation.csv
```

An example of dataset is given in given data/fake_HAPPEI/ directory.

Training and validation sets are described in corresponding *.csv files as follows:
imagefilename1, label2
imagefilename2, label3


CENTRIST features have to be precomputed in advance and placed in data/fake_HAPPEI/CENTRIST directory. One can use CENTRIST code from [here](https://github.com/sometimesfood/spact-matlab) [7], in which CENTRIST descriptor (1D array of length 4096) is computed on non-overlapping 4x4 blocks. Note that code has to be modified to meet the criteria of 4096 features.

Upon initialization, the training process will compute features from given images, which are stored in data/features directory. This is done sequentially:

1) Face bounding boxes. First, faces from images are detected and stored to a file (./data/features/x_bounding_boxes.json)
2) Face happiness intensity. For each detected face (stored in x./data/features/_bounding_boxes.json) face happiness intensity is computed. This is a 6-dimensional vector of probabilities of each happiness intensity. Computed happiness is stored in ./data/features/x_face_happiness.json.
c) Features for the Face-CENTRIST model (./data/features/x_face_features.json). These are 10-dimensional vectors for each face, aggregated in n faces per image (details are given in [2]).

Face Detection model and Face Happiness prediction model are provided as is. If one wants to retrain those models, for face detection we would like to point to original framework used for training  [Tensorbox] (https://github.com/Russell91/TensorBox) [4], whereas code to train VGG16 for face happiness intensity can be found here (TO DO, add link). 


References
----------

  [1]: Abhinav Dhall, Roland Goecke, Jyoti Joshi, Jesse Hoey & Tom Gedeon, EmotiW 2016: Video and Group-level Emotion Recognition Challenges, ACM ICMI 2016. link: https://sites.google.com/site/emotiw2016/challenge-details
  [2]: A. Cerekovic: A deep look into group happiness prediction from images, in Proceedings of the 2016 ACM on International Conference on Multimodal Interaction ICMI'16, preprint
  [3]: http://chenlab.ece.cornell.edu/people/Andy/Andy_files/cvpr09.pdf
  [4]: https://github.com/Russell91/TensorBox
  [5]: https://creativecommons.org/licenses/by-nc/4.0/
  [6]: http://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
  [7]: https://github.com/sometimesfood/spact-matlab
  [8]: https://cs.anu.edu.au/few/Group.htm
