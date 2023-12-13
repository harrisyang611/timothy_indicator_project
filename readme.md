# Indicator Project with Timothy Masters indicators

This is Timothy Master Indicator Project. It uses ML with Timoty Master's indicators to see whether those indicators help the model.

Data are downloaded from yahoo finance, indicators are created through Timothy executables. Price returns are used for modelling. SKlearn is used for a easy replication for other students. Lime and SHAP are used to determine whether the indicators work.

## Slides:

[./slides/2023_feb/slides.pptx](https://github.com/harrisyang611/timothy_indicator_project/blob/master/slides/2023_feb/slides.pptx)


## Requirements:

./req_pkg.txt for pip list
Windows machine is required to run data script.
    ./exe/ includes the pre-compiled Timothy Masters' program,
    For other executables, please see [Github Link](https://github.com/harrisyang611/timothy_master_indicator/tree/main/self_compiled)

## Sample Showcase

./notebook/ has some instruction on how to start on the project.

The sequence you can follow to do for the notebook showcase:
- data_prep_instruction.ipynb [Link Here](https://github.com/harrisyang611/timothy_indicator_project/blob/master/notebook/data_prep_instruction.ipynb)
    This scripts allows you to download the new stock information and create all the indicators.
    requirement: 
        Windows machine
        **VS.txt** file(timothy masters indicator specification) has to be present
- data_with_shap_instruction.ipynb [Link Here](https://github.com/harrisyang611/timothy_indicator_project/blob/master/notebook/model_with_shap_instruction.ipynb)
    This notebook will build the model upon the data created from previous notebook and store the model in joblib format
- construct_equity_plot.ipynb [Link Here](https://github.com/harrisyang611/timothy_indicator_project/blob/master/notebook/construct_equity_plot.ipynb)
    This notebook allows you to construct the equity profile using the data created in first step and model trained in second step.

The data is already run for OEX(S&P 500) and models are built on baseline model / CMMA_10_252, files are store in the ./notebook/testing_data/

Mode details are stated in the notebook respectively.


## Scripts

./src/ for all the codes in script version.
- [Data Prep](https://github.com/harrisyang611/timothy_indicator_project/blob/master/src/data_prep.py)is used to generate data with indicators
- [Model Script](https://github.com/harrisyang611/timothy_indicator_project/blob/master/src/model.py) is used to run main models
- [Model Explain](https://github.com/harrisyang611/timothy_indicator_project/blob/master/src/model_explanation.py) is used to generate model explanation results
