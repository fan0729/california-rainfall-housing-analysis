# california-rainfall-housing-analysis

# California Rainfall and Housing Market Analysis

## Overview

This project investigates the relationship between precipitation patterns and housing market dynamics across California ZIP-code areas from 2015 to 2025. It combines weather data, housing market indicators, and natural language processing (NLP) to examine whether weather-related risk perception provides additional information about housing market behavior.

## Research Questions

1. Is precipitation associated with changes in home values and housing inventory across California?
2. Do housing market responses differ across ZIP-code areas and following extreme rainfall events?
3. Does public sentiment about flooding and property risk improve the explanatory or predictive performance of housing models?

## Data Sources

- **Housing:** Zillow Research — ZIP-level housing value and market indicators.
- **Weather:** Open-Meteo Historical Weather API — historical daily precipitation.
- **Geography:** U.S. Census Bureau — ZIP Code Tabulation Area geographic centroids.
- **Text and sentiment:** Publicly accessible news or social media text, subject to source access and usage requirements.

## Methodology

### 1. Data Engineering
- Collect and clean monthly housing data.
- Aggregate daily precipitation into monthly measures.
- Construct rainfall anomalies relative to historical seasonal baselines.
- Match housing and weather observations by geography and month.

### 2. Exploratory Data Analysis
- Examine regional housing trends and rainfall patterns.
- Visualize precipitation anomalies and extreme rainfall exposure.
- Assess geographic coverage and missing observations.

### 3. Econometric Modeling
- Estimate fixed-effects panel regressions.
- Include ZIP-area and time fixed effects.
- Test lagged precipitation effects.
- Use geographically clustered standard errors.

### 4. NLP and Sentiment Analysis
- Classify text related to flooding, property damage, insurance, and housing risk.
- Construct monthly regional sentiment indicators.
- Compare weather-only and sentiment-enhanced models.

### 5. Model Evaluation
- Evaluate statistical uncertainty and robustness.
- Compare out-of-sample predictive performance.
- Distinguish observational associations from causal effects.

## Tools

Python, pandas, NumPy, GeoPandas, statsmodels, scikit-learn, NLP models, and Tableau.

## Results

Empirical results are pending execution of the data collection and modeling pipeline. Numerical findings will be reported only after the models have been fitted to observed data.

## Limitations

ZIP Code Tabulation Areas approximate USPS ZIP codes. Gridded weather estimates may not capture localized rainfall extremes. Public text may be geographically biased, and sentiment associations do not establish causation.

## Reproducibility

Installation instructions, data requirements, model specifications, and execution steps will be documented in this repository.
