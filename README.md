# Manga-Expression-Tagger--DatasetCreator
This is a model that can classify emotions of manga characters in order to create more dataset for faces in Stable Diffusion LoRA Training for character based expressions. This model is trained on CNN to generate embedding and extraction of features and then trained over Random forest ML model to finally train the main model


Embeddings: (455, 64)
Labels: (455,)

Accuracy: 89.01%

Classification Report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.85      1.00      0.92        11
           2       1.00      0.69      0.82        13
           3       0.74      1.00      0.85        17
           4       1.00      0.62      0.77         8
           5       0.92      1.00      0.96        11
           6       0.94      0.85      0.89        20

    accuracy                           0.89        91
   macro avg       0.92      0.88      0.89        91
weighted avg       0.91      0.89      0.89        91


                        human evaluation 
testimg1.png -> angry   invalid image
testimg2.png -> shock   false
testimg3.png -> shock   false
testimg4.png -> crying  invalid image
testimg5.png -> happy   invalid image
testimg6.png -> shock   correct



This was trained on fairly small dataset (18.26 MB)
### Dataset used:

Mert Köklü - GNU Affero General Public License
https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions
Manga Facial Expressions
Facial expressions of manga (japanese comic) character faces

About Dataset
Context
There are many human facial expression datasets but for mangas (japanese comics) there are none. I hope this tiny dataset would be a good starting point to show the power of manga faces to reflect emotions.

Content
The dataset consists of several emotions (The number next to them is count of it): • Pleased (38) • Angry (54) • Crying (56) • Sad (57) • Embarrassed (67) • Happy (87) • Shock/Surprised (103)


