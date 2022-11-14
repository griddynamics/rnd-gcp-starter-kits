# Price optimization notebook for apparel retail using Google Vertex AI 

## Project description


The major goal of this repository is to represent the minimal set of materials required for the above 
named project to run. The project is described in detail in the following blog-article:

    [Price optimization notebook for apparel retail using Google Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/price-optimization-using-vertex-ai-forecast/)

The goal of the project has been to develop a starter kit for price optmization. According to the article: 

> The starter kit is built upon best practices developed by Grid Dynamics from multiple real-world price optimization projects with leading retail, manufacturing, and CPG customers. It leverages the state-of-the-art AutoML forecasting model from Vertex AI Forecast and shows how to find optimal price points for every product to maximize profit. 

The notebook accompanying the article is available [here](https://colab.research.google.com/drive/1c8uSNi4WV7Aat5pUt5UyC3pNy1P7_ab1) (as on Nov.8, 2022). A slightly polished version of the notebook is allocated here in the folder *./notebooks*.

## Repository structure

- `./notebooks` is the folder with a working copy of the notebook(s)
- `data/input/vk13var-train.csv` is a semi-synthetic training dataset
- `data/input/vk13var-pred.csv`  is the prediction dataset: 4 weeks of historical data (the *context window* data) and 2 weeks of data for prediction horizon with 15 price levels (where the target variable `sales` is represented by NULL values to be replaced by predicted values) 
- `./reports` contains the files produced by the notebook, mainly graphical files with a copy of the graphical output


## Requirements

The notebook is supposed to be run from a Google-drive via Google's Colaboratory (Colab), a free Jupyter notebook environment that runs entirely in the cloud. Besides this, the following requirements are supposed to be satisfied to run the notebook without any modifications:

- Credentials to github repo `https://github.com/griddynamics/rnd-gcp-starter-kits`

- File `requirements.txt` represent the libraries and their versions with which the notebook was tested. This by no means excludes the workability of the notebook with other versions of the libraries.   

One also can use the notebook from other GCP accounts one has credentials for. This would require to change some global variables in 1-2 cells of the notebook and also to upload the two input CSV-files into a GCP bucket. 


## How to setup and run project

- Ensure to meet all requirements
- Upload notebook to your Google drive
- Run the notebook via Google Colab environment

It is reasonabe to run the notebook consecutively, cell-by-cell, paying attention to the comments within the cell, and the output produced for each cell. 

At first run, a pretrained Vertex AI Forecasting model is used. To perform training of a new model in Vertex AI Forecasting, one have to replace the value of the variable `MODEL_NAME` by `None` in the proper cell. 


## License

```
Copyright 2022 Grid Dynamics International, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Authors

- Volodymyr Koliadin, `vkoliadin@griddynamics.com`
- Ilya Katsov, `ikatsov@griddynamics.com`
