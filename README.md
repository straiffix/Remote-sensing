# Remote-sensing
Purpose of this project was training neural network for detection different objects on the satellite images. This was achievied with a help 
of CNN implemented in Keras. 

**Relevant files**

- model.py - building model in Keras
- preparing_ex.py - preprocessing images, creating dataset from samples
- test.py, graphical_test.py, make_predictions_by_classes.py - processing images

**Data**
- Dataset consists of manually extracted satellite images with same resolutions. 
- Example of image:

![Image](/530m_2_copy.png)

- From the images have benn extracted 8 types of objects: cars, crosswalks, separate trees, forests, houses, roads, swimming pools, and samples of lands. 
- Each sample has been preprocessed, from each sample script has created a large amount of different samples. 
- Example of processing:

![Image](/zmienione.png)

**Result**
- In the end whole image is coloured by the colours of each subject. 

![Image](/result7.png)

![Image](/result7_out.png)
