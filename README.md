# Aspect-Based-Sentiment-Analysis
Enhancing Aspect-Based Sentiment Analysis Using Large Language Models And Dependency Parsing

Abstract: Sentiment analysis is a critical task in Natural Language Processing (NLP) that determines sentiment polarity within textual data. Traditional sentiment analysis primarily focuses on binary classification. However, real-world reviews and social media content often exhibit multiple sentiments within a single sentence. This complexity necessitates Aspect-Based Sentiment Analysis (ABSA), which identifies aspect terms and their corresponding sentiments. Despite advancements, existing ABSA models struggle to capture interdependencies between aspect-opinion pairs, leading to misclassifications in multi-aspect scenarios.  To address this, our study proposes enhanced ABSA model which integrates dependency parsing with Large Language Model (LLM)-based learning to incorporate structured semantic knowledge for effective aspect-opinion relationship extraction. The integration of structured feature engineering and domain-specific vocabulary filtering in the proposed work ensures more precise sentiment classification. Experimental evaluations, based on average metrics computed from 5-fold cross validation, demonstrate that the proposed model outperforms existing methods. The model achieves a 3.4% improvement in precision, a 4.9% increase in recall, and a 3.8% boost in F1-score. Additionally, it yields a 5.6% increase in Matthews Correlation Coefficient (MCC), reduces the False Discovery Rate by 3.3%, and lowers the Hamming Loss by 1.7%, ensuring enhanced consistency in multi-aspect sentiment classification. These findings underscore the value of integrating structured semantic knowledge into ABSA, which can significantly enhance the accuracy of sentiment analysis in practical applications.

Dataset:
This project utilizes the SemEval 2014 Task 4 - Aspect Based Sentiment Analysis dataset, specifically the Restaurants domain. The dataset contains customer reviews annotated with aspect terms and their corresponding sentiment polarity. It is sourced from Kaggle and can be accessed here.

Link: https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis?select=Restaurants_Train_v2.csv

Modules:
1. Data pre-processing
2. Data augmentation using LLM
3. Adversarial Sample Generation using LLM
4. Dependancy Parsing using spaCy Model
5. Context based filtering using LLM
6. LLaMA-LoRA fine tuning with integrated K-fold cross validation.
