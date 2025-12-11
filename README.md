# Portfolio-Optimization-GCN-LSTM
GCN, LSTM, and GCN-LSTM based return prediction with EW-MVaR portfolio optimization
# Portfolio Optimization Using GCN, LSTM, GCN-LSTM and EW-MVaR

This repository contains the implementation of stock return prediction using 
GCN, LSTM, and GCN-LSTM models, along with portfolio optimization based on the 
Exponentially Weighted Mean–Value-at-Risk (EW-MVaR) framework.

## Files

- `GCN.py` — Code for the Graph Convolutional Network (GCN) based return prediction.
- `LSTM.py` — Code for the LSTM-based return prediction.
- `GCN-LSTM.py` — Hybrid GCN-LSTM prediction model.
- `EWMVaR.py` — Exponentially Weighted Mean–Value-at-Risk portfolio optimization.

## Dataset

The experiments are conducted on stock market data (e.g., NIFTY 50 or other index constituents).

Due to licensing/size constraints, the dataset itself is not included in this repository.

Users may plug in their own CSV price data following the same format used in the paper.

Note

This code repository is provided as a companion to the research article on
GCN/LSTM-based return prediction and EW-MVaR portfolio optimization.
