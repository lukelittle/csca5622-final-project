# NBA Playoff Predictor

**Lucas Little**  
**CSCA5622**  
**University of Colorado Boulder**

## Overview

This project uses machine learning to predict which NBA teams will qualify for the playoffs. The prediction is based on historical data about team performance, player stats, and injuries. By combining multiple datasets and advanced analysis, the project creates a model that can help forecast playoff outcomes.

## Data Sources

The following datasets were used in this project:
- [NBA Shots Dataset](https://www.kaggle.com/datasets/mexwell/nba-shots): Provides detailed information on shot locations and types.
- [NBA Injury Stats Dataset](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023): Includes historical injury data from 1951-2023.
- [NBA/ABA/BAA Team Statistics Dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats): Contains team performance data, including regular season stats and advanced metrics.

## Steps in the Project

1. **Data Collection and Cleaning**  
   - Gathered data from Kaggle.
   - Cleaned and standardized the data to ensure accuracy and consistency.

2. **Feature Engineering**  
   - Created features such as shooting efficiency, player experience, and injury impact.
   - Added metrics to compare team performance within their conference.

3. **Model Training**  
   - Tested different machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
   - Evaluated models based on accuracy, precision, and recall.

4. **Results**  
   - The Logistic Regression model performed the best, achieving 83.3% accuracy on test data.
   - This model is now ready for further tuning and deployment.

## Key Takeaways

- Combining team, player, and injury data improves prediction accuracy.
- Logistic Regression is a simple yet effective method for predicting playoff outcomes.
- The project framework is reproducible and can be extended for future analyses.

## Next Steps

- Train additional models and perform hyperparameter tuning for improved accuracy.
- Analyze feature importance to gain insights into what drives playoff success.
- Deploy the model to predict outcomes for upcoming NBA seasons.

## Keywords

NBA, machine learning, playoff prediction, data analysis
