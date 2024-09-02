# Conversion Triple Fit

This repository provides tools for fitting models to datasets using triplet data structures. It includes integration with MATLAB for specific computational tasks.

## Installation

### Requirements

To install the required dependencies, use:

```bash
pip install -r requirements.txt
```

### Using MATLAB

If you want to use the original MATLAB engine, follow these steps:

1. **Installing the MATLAB Library**

   - First, find your MATLAB root folder by opening MATLAB and running the command `matlabroot`. This command will give you the path to the root folder of MATLAB.
   
   - Next, open your terminal. If you are using Windows, you can do this by pressing `Windows + R`, typing `cmd`, and pressing `Enter`.
   
   - In the terminal, navigate to the MATLAB Python engine directory by running the following command (make sure to replace `matlabroot` with the path you found):

     ```bash
     cd matlabroot\extern\engines\python
     ```

   - Finally, run the following command to install the MATLAB Python library:

     ```bash
     python3 setup.py install
     ```

   This will install the MATLAB Python engine and allow you to run MATLAB code from within your Python environment.

### Running the Code

To fit models, use the `fit_all` function from the `triplet_fitting.fit` module.

### Dependencies

- pandas
- numpy
- scipy
- sklearn
- tqdm
- matlab.engine
- matplotlib
- seaborn
- plotly
