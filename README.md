## Stock Prediction with Textual and Numerical Analysis

This repository documents experiments in forecasting stock prices using news headline sentiment.
Nine machine learning models are compared, with only the best model selected for tuning and testing.

### Approach

1. **Data Cleaning**: Clean the news headline dataset by removing special characters and punctuation. Similarly, preprocess the stock price dataset by filling in missing values using interpolation.
2. **Exploratory Data Analysis (EDA)**: Conduct EDA to understand data characteristics and identify trends. For news headlines, create a word cloud to visualize frequently used words.
For stock prices, analyze the distribution of the 'Adj Close' column and plot a line chart of stock prices between January 2, 2018, and May 10, 2024.
3. **Sentiment Analysis**: Apply the VADER sentiment analysis tool to news headlines.
This process generates 'polarity', 'subjectivity', 'compound', 'positive', 'negative', and 'neutral' scores, which will be used as predictors.
4. **Data Preparation**: Combine the sentiment scores ('polarity', 'subjectivity', 'compound', 'positive', 'negative', 'neutral') with the 'Adj Close' stock price as the final dataset for model training.
5. **Data Splitting**: Divide the dataset into training and testing sets.
6. **Model Training**: Train various baseline regression models and evaluate their performance using Average RMSE with cv=5.
7. **Model Tuning**: Fine-tune the best-performing baseline model.
8. **Model Evaluation**: Evaluate the performance of the fine-tuned model using RMSE and visualize the results.

### Result

In this experiment, here's the model training result of the 9 baseline model:
| **ML Model** | **AVG RMSE Score using cv=5** | **STD** |
| --- | --- | -- |
| LGBM | 24.2174 | 0.3526 |
| Extra Tree | 24.4418 | 0.2534 |
| Random Forest RMSE | 23.9594 | 0.2565 |
| Gradient Boosting RMSE | 23.6425 | 0.2741 |
| KNeighbors RMSE | 31.6611 | 0.5431 |
| Decision Tree | 28.3030 | 0.8739 |
| XGB | 25.2187 | 0.2184 |
| Cat Boost | 23.8762 | 0.4318 |
| Linear Regression | 24.1302 | 0.3238 |

Across 9 models, GBR-baseline outperformed others. Here's the best parameter for the GBR, with this configuration the average CV RMSE was reduced to 23.568.
`
    'learning_rate': 0.05416033402541074,
    'max_depth': 7,
    'max_features': 'sqrt',
    'min_samples_leaf': 20,
    'min_samples_split': 20,
    'subsample': 0.5
`

### Future Work
Although we have identified the best model thus far, the prediction errors are still significant.
Further research is required to explore different configurations for the predictor and model.
