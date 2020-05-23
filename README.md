# Video Action Recognition using Deep Learning





## <u>Project Structure</u>

* The data goes inside the `input` folder. **Get the [data](https://drive.google.com/file/d/107w498Ahs7hxuCAd8tEX0u33OQUYWq4Y/view) from [here](https://drive.google.com/file/d/107w498Ahs7hxuCAd8tEX0u33OQUYWq4Y/view)**.

```
├───input
│   ├───data
│   │   ├───badminton
│   │   ├───baseball
│   │   ├───basketball
│   │   ├───boxing
│   │   ...
│   └───example_clips
│           basketball.mp4
│           boxing1.mp4
│           chess.mp4
|    |   data.csv 
├───outputs
└───src
    │   cnn_models.py
    │   prepare_data.py
    │   test.py
    │   train.py
```

* You can extract the `data.zip` file inside the `input` folder and you will get all the subfolders containing the sports images according to the categories.
* `input` folder also contains the `example_clips` subfolder that contains the short video clips that we will test our trained deep learning model on.
* `outputs` folder will contain all the output files. These include the loss and accuracy graph plots, the trained model, and some other files that we will discover as we move further.
* `src` contains all the python scripts.
  - `prepare_data.py`: To prepare the dataset for the training images and the `data.csv` file.
  - `cnn_models.py`: Contains the neural network model.
  - `train.py`: Contains the training and validation scripts.
  - `test.py`: This python file is for testing on the trained neural network model on the `example_clips` videos.



* ***Note:*** The trained mode, that is, `model.pth` inside the `outputs` folder in this repository has been trained on `basketball`,  `boxing`, and `chess` data. You can easily generate the `data.csv` file for other images as well. 
  * *Take a look inside the `prepare_data.py` script. Just add more image folder names to the `create_labels` list and execute the script.*
  * *Then run `train.py` to train on those image data as well.*  



## <u>References</u>

* Drawn inspiration from [this](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/) post by Adrian Rosebrock on [PyImageSearch](https://www.pyimagesearch.com/).
  * The original post used Keras and ResNet-50 transfer learning.
  * **I have used PyTorch, and a custom CNN deep learning model. Take a look at `cnn_models.py` inside `src` folder to get more insights.**

* [Retrieving actions in movies](http://www.irisa.fr/vista/Papers/2007_iccv_laptev.pdf), **Ivan Laptev and Patrick Perez**.
* [Learning realistic human actions from movies](https://www.irisa.fr/vista/Papers/2008_cvpr_laptev.pdf), **Laptev et al**.
* [Finding Actors and Actions in Movies](https://www.di.ens.fr/willow/research/actoraction/).
* [An End-to-end 3D Convolutional Neural Network for Action Detection and Segmentation in Videos](https://www.academia.edu/35318871/An_End-to-end_3D_Convolutional_Neural_Network_for_Action_Detection_and_Segmentation_in_Videos?email_work_card=title), **Chen Chen**.
* Evaluation of local spatio-temporal features for action recognition, Wang et al.         
* Recognizing Realistic Actions from Videos “in the Wild”, Liu et al. 
* A Text Retrieval Approach to Object Matching in Videos, **Sivic and Zisserman**. 
* [Modeling Temporal Structure of Decomposable Motion Segments for Activity Classification](https://www.researchgate.net/profile/Juan_Carlos_Niebles/publication/221304534_Modeling_Temporal_Structure_of_Decomposable_Motion_Segments_for_Activity_Classification/links/00b495304fe61c9551000000.pdf), **Juan Carlos Niebles1,2,3, Chih-Wei Chen1, and Li Fei-Fei1**
* [Sequential Deep Learning for Human Action Recognition](), **Baccouche et al**.
* [Histograms of Oriented Gradients for Human Detection,](https://hal.inria.fr/file/index/docid/548512/filename/hog_cvpr2005.pdf)  **Navneet Dalal, Bill Triggs**.
* [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/pdf/1212.0402.pdf), **Khurram Soomro, Amir Roshan Zamir and Mubarak Shah**.
* [DELVING DEEPER INTO CONVOLUTIONAL NETWORKS FOR LEARNING VIDEO REPRESENTATIONS](https://arxiv.org/pdf/1511.06432v4.pdf)
* [DEEP MULTI-SCALE VIDEO PREDICTION BEYOND MEAN SQUARE ](https://arxiv.org/pdf/1511.05440v6.pdf)
* Credits:
  * `basketball.mp4`:
    * Link: https://www.pexels.com/video/athletes-warming-up-1585619/.
  * `chess.mp4`: 
    * Link: https://www.pexels.com/video/queen-check-in-chess-855386/.
  * `boxing1.mp4`: 
    * Link: https://www.youtube.com/watch?v=3kR-w73tUWg.
  * 

