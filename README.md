# README #

This README documents the necessary steps to set up the running environment for Kalstra.

# Kalstra
*A Novel Hybrid Architecture Integrating KAN and Sequential-Transformer for Robust Cell Type Annotation in Cross-Species Transcriptomics*

## Environment Configuration
The code runs under **Python 3.9.19** and **Tensorflow 2.15.0**. Create a Tensorflow environment and install required packages, such as "numpy", "pandas", "keras" and "scanpy" .
Please refer to requirements.txt for more details.

### Installation:
```python
pip install -r requirements.txt
```
## Model Training and Testing
* After setting up the above files, execute Python files sequentially in the folder **KALSTRA**.

### Output Files
* Training results are saved into **modelsave/epoch200.txt**

## Code Structure

```python
LSKAN.py:      ATLSTM layer and CFKAN model
GMHA.py:       Attention layer and BAFFN layer
model.py:      Integrate LSKAN and GMHA into KALSTRA
training.py:   Model training
testing.py:    Prediction results of dataset
```

## Contact Information
Have any questions or issues related to the repository, please contact Dr. Binhua Tang (bh.tang@hhu.edu.cn) or Yiyao Chen (221620010005@hhu.edu.cn).
