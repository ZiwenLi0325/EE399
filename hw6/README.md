# EE399
EE399 Homework submission
# Homework Set 6: Analysis of SHRED Model Performance on Sea-Surface Temperature Data

Author: Ziwen(https://github.com/ZiwenLi0325)

## Abstract:
This study explores the impacts of varying time lag, noise, and number of sensors on the performance of a SHRED model trained on sea-surface temperature data. This analysis provides insights into how the model responds to different input conditions, which could guide future data collection strategies and model tuning efforts.


## Sec. I. Introduction and Overview:
The application of machine learning algorithms to predict sea-surface temperatures is a vital task in the domain of climatology. This report investigates the application of an LSTM/decoder model, specifically the SHRED model, to this problem. We assess the performance of the model under different conditions, examining the effects of time lag, the introduction of Gaussian noise, and the number of sensors used.

![recon](recon.png "Reconstruction image of SST")


## Sec. II. Theoretical Background:
The SHRED model used in this study is an LSTM-based model which incorporates an encoder-decoder structure. LSTM networks are well-suited for time-series prediction tasks due to their ability to capture long-term dependencies in the data. The performance of such models can be affected by various factors, including the length of the time lag used, the amount of noise in the data, and the number of sensors used to collect the data.

## Sec. III. Algorithm Implementation and Development:

The LSTM/decoder model was implemented in Python using PyTorch. Training, validation, and testing datasets were created from the sea-surface temperature data. To explore the impact of time lag, noise, and number of sensors on model performance, the model was trained and evaluated under different conditions.

## Sec. IV. Computational Results

Our results showed that the model's performance varied with changes in time lag, noise, and the number of sensors. Notably, an increase in noise tended to decrease model performance, while the impact of changing the time lag and the number of sensors was more complex and depended on the specific configuration of these variables.

## V. Summary and Conclusions

This study provides valuable insights into the performance characteristics of the SHRED model on sea-surface temperature prediction tasks. It highlights the impact of various factors on model performance, which could guide future work in data collection and model tuning. Future work could further explore these factors and investigate additional ways to improve the performance of LSTM-based models on this task.
