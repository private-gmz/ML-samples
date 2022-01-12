# samples of Machine Learning projects
This repository contains some machine learning code samples (MATLAB files and notbooks/Python).
- age-gender-race multi-model (notebook/tensorflow): 
 This code builds a multi_task model to predict (age, gender and race) using utkface dataset which is available on kaggle. The dataset containes approximately 20,000 images with three labels (age, gender and race).
 The purpose of this notebook is to try many things regarding face detection like (transfer learning, image generating, augmentation, one model with three branches for age, race and gender).
 The model is a prototype trained only for 20 epochs and needs to be tuned.
- emotion folder (python files): a flask app that uses a trained machine learning model to detect the seven basic emotions in realtime using webcam then scrap the top rated movies of a specific genre according to the detected feeling.
- heart disease (notebook/tensorflow): build and train a model using heart disease dataset on kaggle (contains patients history with five classes that indicate the absence(0) or presence of heart disease in four stages(1,2,3,4)).
the model uses SMOTE technique to solve the "class imbalance" problem in the dataset. This technique increased the model accuracy from 60% to more than 95%.
- idea generating (notebook/tensorflow): this contains code for scraping Artificial Intelligence project titles from stackoverflow answers on (projects suggestions) and Stanford University projects library, then using pretrained models like (GPT-2 and textgenrnn) to generate new ideas.
- loan status (notebook/tensorflow): explore and process loan status dataset (available on kaggle) then build and train a model (only 50 epochs) using k-fold cross-validation to classify if the applicant deserves the loan or not based on his data.
- next steps prediction (MATLAB): This code is to predict the next 4 steps of a user mobility in a MANET network that covers 500 m^2. I buid a NARXNET neural network that uses weights calculated from ELM neural network as initial weights. NARXNET inputs are time series coordinates (X, Y) of ¬100 users and variance value of user steps in a specific previous period.
