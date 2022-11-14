# Churn Predictions for Telco company Using Google Vertex AI

This notebook represents a starter kit for implementing churn analytic pipeline with autoML provided by Vertex AI. In other words, it shows best practices of how to implement auto ml capabilities in modeling churn predictions, and how to utilise those models for gaining better insights when it comes to reducing number of churned users. Considering all above, we included following steps in the notebook:

1. Initialization and authentification
2. Usuful functions
2. Load data
3. Short-term churn prediction - model used for predicting churns for short terms time horizon, such as prediction for next month. In general, it shows **WHO** is likely going to churn. This type of models, used the most recent behavioral data to predict next user event.
4. Long-term churn predictor - model is used to revield **WHEN** users have more chance to churn, and it helps business retaining them on a time. This model use sequence of events from recent users past in order to predict their future behaviour, usually inside predefined future time horizont.
5. Uplift modeling for selecting treatment - model that shows **HOW** we may prevent users from churning with a best treatments suited for them. Uplift models help buisness in matching users with the right treatment and it could be used in process of optimisation of costs.
6. Identifiaction of triggers - it shows **WHAT** is triggering our users to churn. This type of analyses is capable of showing business where is the bottlenecks of the system.

Models from this notebook should be implement together by following next steps:
- step 1: Implement account and interaction data to train long & short term models that will revield users that have higher probability of churning from services.
- step 2: Implement uplift model as shown in this notebook, to estimate what is best strategy for retaining individual user, and use output of the model to optimise company retain budget.
- step 3: Utilise user generated data to extract churn triggers in a system. 

The notebook is supposed to be run by Google Collaboratory.

## Initialization and authentification

In this section, we perfom basic initialization and set some variables.


```python
#
# Reinstall google-cloud-aiplatform
#

!pip3 uninstall -y google-cloud-aiplatform
!pip3 install google-cloud-aiplatform
 
import IPython
 
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

    [33mWARNING: Skipping google-cloud-aiplatform as it is not installed.[0m
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting google-cloud-aiplatform
      Downloading google_cloud_aiplatform-1.18.3-py2.py3-none-any.whl (2.3 MB)
    [K     |████████████████████████████████| 2.3 MB 5.1 MB/s 
    [?25hCollecting protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5
      Downloading protobuf-4.21.9-cp37-abi3-manylinux2014_x86_64.whl (408 kB)
    [K     |████████████████████████████████| 408 kB 45.1 MB/s 
    [?25hCollecting google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0
      Downloading google_api_core-2.10.2-py3-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 53.3 MB/s 
    [?25hCollecting google-cloud-resource-manager<3.0.0dev,>=1.3.3
      Downloading google_cloud_resource_manager-1.6.3-py2.py3-none-any.whl (233 kB)
    [K     |████████████████████████████████| 233 kB 61.1 MB/s 
    [?25hCollecting google-cloud-storage<3.0.0dev,>=1.32.0
      Downloading google_cloud_storage-2.6.0-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 54.8 MB/s 
    [?25hCollecting proto-plus<2.0.0dev,>=1.22.0
      Downloading proto_plus-1.22.1-py3-none-any.whl (47 kB)
    [K     |████████████████████████████████| 47 kB 4.4 MB/s 
    [?25hRequirement already satisfied: google-cloud-bigquery<3.0.0dev,>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from google-cloud-aiplatform) (1.21.0)
    Requirement already satisfied: packaging<22.0.0dev,>=14.3 in /usr/local/lib/python3.7/dist-packages (from google-cloud-aiplatform) (21.3)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (2.23.0)
    Requirement already satisfied: google-auth<3.0dev,>=1.25.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (1.35.0)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /usr/local/lib/python3.7/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (1.56.4)
    Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /usr/local/lib/python3.7/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (1.50.0)
    Collecting grpcio-status<2.0dev,>=1.33.2
      Downloading grpcio_status-1.50.0-py3-none-any.whl (14 kB)
    Requirement already satisfied: setuptools>=40.3.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (57.4.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (4.9)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (1.15.0)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (4.2.4)
    Requirement already satisfied: google-cloud-core<2.0dev,>=1.0.3 in /usr/local/lib/python3.7/dist-packages (from google-cloud-bigquery<3.0.0dev,>=1.15.0->google-cloud-aiplatform) (1.0.3)
    Requirement already satisfied: google-resumable-media!=0.4.0,<0.5.0dev,>=0.3.1 in /usr/local/lib/python3.7/dist-packages (from google-cloud-bigquery<3.0.0dev,>=1.15.0->google-cloud-aiplatform) (0.4.1)
    Collecting google-cloud-core<2.0dev,>=1.0.3
      Downloading google_cloud_core-1.7.3-py2.py3-none-any.whl (28 kB)
    Collecting grpc-google-iam-v1<1.0.0dev,>=0.12.4
      Downloading grpc_google_iam_v1-0.12.4-py2.py3-none-any.whl (26 kB)
    Collecting google-cloud-storage<3.0.0dev,>=1.32.0
      Downloading google_cloud_storage-2.5.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 47.9 MB/s 
    [?25h  Downloading google_cloud_storage-2.4.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 44.8 MB/s 
    [?25h  Downloading google_cloud_storage-2.3.0-py2.py3-none-any.whl (107 kB)
    [K     |████████████████████████████████| 107 kB 8.3 MB/s 
    [?25h  Downloading google_cloud_storage-2.2.1-py2.py3-none-any.whl (107 kB)
    [K     |████████████████████████████████| 107 kB 47.4 MB/s 
    [?25h  Downloading google_cloud_storage-2.2.0-py2.py3-none-any.whl (107 kB)
    [K     |████████████████████████████████| 107 kB 45.6 MB/s 
    [?25h  Downloading google_cloud_storage-2.1.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 47.0 MB/s 
    [?25h  Downloading google_cloud_storage-2.0.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 46.8 MB/s 
    [?25h  Downloading google_cloud_storage-1.44.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 40.2 MB/s 
    [?25h  Downloading google_cloud_storage-1.43.0-py2.py3-none-any.whl (106 kB)
    [K     |████████████████████████████████| 106 kB 46.5 MB/s 
    [?25h  Downloading google_cloud_storage-1.42.3-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 44.6 MB/s 
    [?25h  Downloading google_cloud_storage-1.42.2-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 45.8 MB/s 
    [?25h  Downloading google_cloud_storage-1.42.1-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 48.1 MB/s 
    [?25h  Downloading google_cloud_storage-1.42.0-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 42.1 MB/s 
    [?25h  Downloading google_cloud_storage-1.41.1-py2.py3-none-any.whl (105 kB)
    [K     |████████████████████████████████| 105 kB 46.7 MB/s 
    [?25h  Downloading google_cloud_storage-1.41.0-py2.py3-none-any.whl (104 kB)
    [K     |████████████████████████████████| 104 kB 45.8 MB/s 
    [?25h  Downloading google_cloud_storage-1.40.0-py2.py3-none-any.whl (104 kB)
    [K     |████████████████████████████████| 104 kB 38.1 MB/s 
    [?25h  Downloading google_cloud_storage-1.39.0-py2.py3-none-any.whl (103 kB)
    [K     |████████████████████████████████| 103 kB 45.2 MB/s 
    [?25h  Downloading google_cloud_storage-1.38.0-py2.py3-none-any.whl (103 kB)
    [K     |████████████████████████████████| 103 kB 46.4 MB/s 
    [?25h  Downloading google_cloud_storage-1.37.1-py2.py3-none-any.whl (103 kB)
    [K     |████████████████████████████████| 103 kB 47.1 MB/s 
    [?25h  Downloading google_cloud_storage-1.37.0-py2.py3-none-any.whl (103 kB)
    [K     |████████████████████████████████| 103 kB 43.9 MB/s 
    [?25h  Downloading google_cloud_storage-1.36.2-py2.py3-none-any.whl (97 kB)
    [K     |████████████████████████████████| 97 kB 5.5 MB/s 
    [?25h  Downloading google_cloud_storage-1.36.1-py2.py3-none-any.whl (97 kB)
    [K     |████████████████████████████████| 97 kB 5.5 MB/s 
    [?25h  Downloading google_cloud_storage-1.36.0-py2.py3-none-any.whl (97 kB)
    [K     |████████████████████████████████| 97 kB 3.8 MB/s 
    [?25h  Downloading google_cloud_storage-1.35.1-py2.py3-none-any.whl (96 kB)
    [K     |████████████████████████████████| 96 kB 2.2 MB/s 
    [?25h  Downloading google_cloud_storage-1.35.0-py2.py3-none-any.whl (96 kB)
    [K     |████████████████████████████████| 96 kB 3.1 MB/s 
    [?25h  Downloading google_cloud_storage-1.34.0-py2.py3-none-any.whl (96 kB)
    [K     |████████████████████████████████| 96 kB 1.3 MB/s 
    [?25h  Downloading google_cloud_storage-1.33.0-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 5.6 MB/s 
    [?25h  Downloading google_cloud_storage-1.32.0-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 7.6 MB/s 
    [?25hINFO: pip is looking at multiple versions of google-cloud-resource-manager to determine which version is compatible with other requirements. This could take a while.
    Collecting google-cloud-resource-manager<3.0.0dev,>=1.3.3
      Downloading google_cloud_resource_manager-1.6.2-py2.py3-none-any.whl (233 kB)
    [K     |████████████████████████████████| 233 kB 45.8 MB/s 
    [?25h  Downloading google_cloud_resource_manager-1.6.1-py2.py3-none-any.whl (231 kB)
    [K     |████████████████████████████████| 231 kB 47.6 MB/s 
    [?25h  Downloading google_cloud_resource_manager-1.6.0-py2.py3-none-any.whl (231 kB)
    [K     |████████████████████████████████| 231 kB 66.2 MB/s 
    [?25hCollecting protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5
      Downloading protobuf-3.20.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
    [K     |████████████████████████████████| 1.0 MB 59.7 MB/s 
    [?25hCollecting google-cloud-resource-manager<3.0.0dev,>=1.3.3
      Downloading google_cloud_resource_manager-1.5.1-py2.py3-none-any.whl (230 kB)
    [K     |████████████████████████████████| 230 kB 54.5 MB/s 
    [?25h  Downloading google_cloud_resource_manager-1.5.0-py2.py3-none-any.whl (230 kB)
    [K     |████████████████████████████████| 230 kB 46.8 MB/s 
    [?25h  Downloading google_cloud_resource_manager-1.4.1-py2.py3-none-any.whl (402 kB)
    [K     |████████████████████████████████| 402 kB 49.2 MB/s 
    [?25h  Downloading google_cloud_resource_manager-1.4.0-py2.py3-none-any.whl (402 kB)
    [K     |████████████████████████████████| 402 kB 46.2 MB/s 
    [?25hINFO: pip is looking at multiple versions of google-cloud-resource-manager to determine which version is compatible with other requirements. This could take a while.
      Downloading google_cloud_resource_manager-1.3.3-py2.py3-none-any.whl (286 kB)
    [K     |████████████████████████████████| 286 kB 50.4 MB/s 
    [?25hINFO: pip is looking at multiple versions of google-cloud-core to determine which version is compatible with other requirements. This could take a while.
    Collecting google-cloud-core<2.0dev,>=1.0.3
      Downloading google_cloud_core-1.7.2-py2.py3-none-any.whl (28 kB)
      Downloading google_cloud_core-1.7.1-py2.py3-none-any.whl (28 kB)
      Downloading google_cloud_core-1.7.0-py2.py3-none-any.whl (28 kB)
      Downloading google_cloud_core-1.6.0-py2.py3-none-any.whl (28 kB)
      Downloading google_cloud_core-1.5.0-py2.py3-none-any.whl (27 kB)
      Downloading google_cloud_core-1.4.4-py2.py3-none-any.whl (27 kB)
      Downloading google_cloud_core-1.4.3-py2.py3-none-any.whl (27 kB)
      Downloading google_cloud_core-1.4.2-py2.py3-none-any.whl (26 kB)
      Downloading google_cloud_core-1.4.1-py2.py3-none-any.whl (26 kB)
      Downloading google_cloud_core-1.4.0-py2.py3-none-any.whl (26 kB)
      Downloading google_cloud_core-1.3.0-py2.py3-none-any.whl (26 kB)
      Downloading google_cloud_core-1.2.0-py2.py3-none-any.whl (26 kB)
      Downloading google_cloud_core-1.1.0-py2.py3-none-any.whl (26 kB)
    INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
    Collecting google-cloud-bigquery<3.0.0dev,>=1.15.0
      Downloading google_cloud_bigquery-2.34.4-py2.py3-none-any.whl (206 kB)
    [K     |████████████████████████████████| 206 kB 47.3 MB/s 
    [?25hCollecting google-cloud-core<3.0.0dev,>=1.4.1
      Downloading google_cloud_core-2.3.2-py2.py3-none-any.whl (29 kB)
    Collecting google-resumable-media<3.0dev,>=0.6.0
      Downloading google_resumable_media-2.4.0-py2.py3-none-any.whl (77 kB)
    [K     |████████████████████████████████| 77 kB 4.1 MB/s 
    [?25hRequirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /usr/local/lib/python3.7/dist-packages (from google-cloud-bigquery<3.0.0dev,>=1.15.0->google-cloud-aiplatform) (2.8.2)
    Collecting google-crc32c<2.0dev,>=1.0
      Downloading google_crc32c-1.5.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (32 kB)
    Collecting grpcio-status<2.0dev,>=1.33.2
      Downloading grpcio_status-1.49.1-py3-none-any.whl (14 kB)
      Downloading grpcio_status-1.48.2-py3-none-any.whl (14 kB)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging<22.0.0dev,>=14.3->google-cloud-aiplatform) (3.0.9)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3.0dev,>=1.25.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (0.4.8)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (2022.9.24)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0dev,>=1.32.0->google-cloud-aiplatform) (1.24.3)
    Installing collected packages: protobuf, grpcio-status, google-crc32c, google-api-core, proto-plus, grpc-google-iam-v1, google-resumable-media, google-cloud-core, google-cloud-storage, google-cloud-resource-manager, google-cloud-bigquery, google-cloud-aiplatform
      Attempting uninstall: protobuf
        Found existing installation: protobuf 3.17.3
        Uninstalling protobuf-3.17.3:
          Successfully uninstalled protobuf-3.17.3
      Attempting uninstall: google-api-core
        Found existing installation: google-api-core 1.31.6
        Uninstalling google-api-core-1.31.6:
          Successfully uninstalled google-api-core-1.31.6
      Attempting uninstall: google-resumable-media
        Found existing installation: google-resumable-media 0.4.1
        Uninstalling google-resumable-media-0.4.1:
          Successfully uninstalled google-resumable-media-0.4.1
      Attempting uninstall: google-cloud-core
        Found existing installation: google-cloud-core 1.0.3
        Uninstalling google-cloud-core-1.0.3:
          Successfully uninstalled google-cloud-core-1.0.3
      Attempting uninstall: google-cloud-storage
        Found existing installation: google-cloud-storage 1.18.1
        Uninstalling google-cloud-storage-1.18.1:
          Successfully uninstalled google-cloud-storage-1.18.1
      Attempting uninstall: google-cloud-bigquery
        Found existing installation: google-cloud-bigquery 1.21.0
        Uninstalling google-cloud-bigquery-1.21.0:
          Successfully uninstalled google-cloud-bigquery-1.21.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.9.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    tensorboard 2.9.1 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    pandas-gbq 0.13.3 requires google-cloud-bigquery[bqstorage,pandas]<2.0.0dev,>=1.11.1, but you have google-cloud-bigquery 2.34.4 which is incompatible.
    google-cloud-translate 1.5.0 requires google-api-core[grpc]<2.0.0dev,>=1.6.0, but you have google-api-core 2.10.2 which is incompatible.
    google-cloud-translate 1.5.0 requires google-cloud-core<2.0dev,>=1.0.0, but you have google-cloud-core 2.3.2 which is incompatible.
    google-cloud-language 1.2.0 requires google-api-core[grpc]<2.0.0dev,>=1.6.0, but you have google-api-core 2.10.2 which is incompatible.
    google-cloud-firestore 1.7.0 requires google-api-core[grpc]<2.0.0dev,>=1.14.0, but you have google-api-core 2.10.2 which is incompatible.
    google-cloud-firestore 1.7.0 requires google-cloud-core<2.0dev,>=1.0.3, but you have google-cloud-core 2.3.2 which is incompatible.
    google-cloud-datastore 1.8.0 requires google-api-core[grpc]<2.0.0dev,>=1.6.0, but you have google-api-core 2.10.2 which is incompatible.
    google-cloud-datastore 1.8.0 requires google-cloud-core<2.0dev,>=1.0.0, but you have google-cloud-core 2.3.2 which is incompatible.
    firebase-admin 4.4.0 requires google-api-core[grpc]<2.0.0dev,>=1.14.0; platform_python_implementation != "PyPy", but you have google-api-core 2.10.2 which is incompatible.[0m
    Successfully installed google-api-core-2.10.2 google-cloud-aiplatform-1.18.3 google-cloud-bigquery-2.34.4 google-cloud-core-2.3.2 google-cloud-resource-manager-1.6.3 google-cloud-storage-2.6.0 google-crc32c-1.5.0 google-resumable-media-2.4.0 grpc-google-iam-v1-0.12.4 grpcio-status-1.48.2 proto-plus-1.22.1 protobuf-3.20.3







    {'status': 'ok', 'restart': True}




```python
!pip install waterfallcharts
!pip install pyyaml
!python -m spacy download en_core_web_md 
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting waterfallcharts
      Downloading waterfallcharts-3.8.tar.gz (3.9 kB)
    Building wheels for collected packages: waterfallcharts
      Building wheel for waterfallcharts (setup.py) ... [?25l[?25hdone
      Created wheel for waterfallcharts: filename=waterfallcharts-3.8-py3-none-any.whl size=3414 sha256=00ecff046ccddf97154abf325103240cdb693611b5438811b4abbc321e5a4b62
      Stored in directory: /root/.cache/pip/wheels/4a/b2/85/9b6eda27fab2effa580c6942fb094e8536cc46e8741ec5f586
    Successfully built waterfallcharts
    Installing collected packages: waterfallcharts
    Successfully installed waterfallcharts-3.8
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (6.0)
    2022-11-10 11:56:50.739326: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting en-core-web-md==3.4.1
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.1/en_core_web_md-3.4.1-py3-none-any.whl (42.8 MB)
    [K     |████████████████████████████████| 42.8 MB 1.7 MB/s 
    [?25hRequirement already satisfied: spacy<3.5.0,>=3.4.0 in /usr/local/lib/python3.7/dist-packages (from en-core-web-md==3.4.1) (3.4.2)
    Requirement already satisfied: thinc<8.2.0,>=8.1.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (8.1.5)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.23.0)
    Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (1.21.6)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (4.64.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.11.3)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.0.8)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (1.10.2)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (57.4.0)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (1.0.3)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.0.8)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.3.0)
    Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (0.4.2)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.0.7)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.10 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.0.10)
    Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (0.6.2)
    Requirement already satisfied: wasabi<1.1.0,>=0.9.1 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (0.10.1)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (21.3)
    Requirement already satisfied: typing-extensions<4.2.0,>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (4.1.1)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (1.0.9)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.10.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.0.9)
    Requirement already satisfied: smart-open<6.0.0,>=5.2.1 in /usr/local/lib/python3.7/dist-packages (from pathy>=0.3.5->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (5.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (3.0.4)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.7/dist-packages (from thinc<8.2.0,>=8.1.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (0.0.3)
    Requirement already satisfied: blis<0.8.0,>=0.7.8 in /usr/local/lib/python3.7/dist-packages (from thinc<8.2.0,>=8.1.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (0.7.9)
    Requirement already satisfied: click<9.0.0,>=7.1.1 in /usr/local/lib/python3.7/dist-packages (from typer<0.5.0,>=0.3.0->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (7.1.2)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->spacy<3.5.0,>=3.4.0->en-core-web-md==3.4.1) (2.0.1)
    Installing collected packages: en-core-web-md
    Successfully installed en-core-web-md-3.4.1
    [38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_md')



```python
#
# Authentication
#

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```


```python
! echo https://{TOKEN}@github.com/griddynamics/rnd-gcp-starter-kits/
```

    https://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@github.com/griddynamics/rnd-gcp-starter-kits/



```python
#
# Pull starting kit folder from github repo
#

TOKEN = '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'
!mkdir rnd-gcp-starter-kits && \
  cd rnd-gcp-starter-kits && \
  git init && \
  git config core.sparseCheckout true && \
  git remote add -f origin  https://{TOKEN}@github.com/griddynamics/rnd-gcp-starter-kits/ && \
  echo churn-prevention-vertex-ai >> .git/info/sparse-checkout && \
  git pull origin main
```

    Initialized empty Git repository in /content/rnd-gcp-starter-kits/.git/
    Updating origin
    remote: Enumerating objects: 59, done.[K
    remote: Counting objects: 100% (59/59), done.[K
    remote: Compressing objects: 100% (55/55), done.[K
    remote: Total 59 (delta 7), reused 26 (delta 3), pack-reused 0
    Unpacking objects: 100% (59/59), done.
    From https://github.com/griddynamics/rnd-gcp-starter-kits
     * [new branch]      main       -> origin/main
    From https://github.com/griddynamics/rnd-gcp-starter-kits
     * branch            main       -> FETCH_HEAD



```python
#
# Load configuration file, if needed than 
#

import yaml

config = yaml.safe_load(open(f'rnd-gcp-starter-kits/churn-prevention-vertex-ai/config.yml'))
```


```python
#
# Set some global variables with your project details and other information:
#

from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H%M%S")

REGION = config['gcp']['region']
PROJECT_ID = config['gcp']['project_id']
BUCKET_NAME = config['gcp']['bucket_name']

print(TIMESTAMP)
```

    2022_11_10_122044



```python
#
# Create project, bucket and upload files 
#

# # check if project and bucket exist if no create it
# find_project = !gcloud projects list --filter $PROJECT_ID
# if 'Listed 0 items.' in find_project:
#    !gcloud projects create {PROJECT_ID}

# # set project
# !gcloud config set project {PROJECT_ID}

# # check if bucket exist
# from google.cloud import storage
#
# client = storage.Client()
# try:
#     bucket = client.get_bucket(BUCKET_NAME)
# except:
#     !gsutil mb gs://{BUCKET_NAME}

# # transfer files to bucket
# for key, file in config['data']['input_data'].items():
#     if 'file_name' in key:
#         !gsutil cp data/{file} gs://{BUCKET_NAME}/
#     elif 'file_path' in key:
#         !gsutil cp {file} gs://{BUCKET_NAME}/
#     print("Copy {file} to {BUCKET_NAME}")
```


```python
#
# Initialize AI platform:
#

from google.cloud import aiplatform
import os

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_NAME)
os.environ["GCLOUD_PROJECT"] = PROJECT_ID 
```


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## Useful functions


```python
#
# Function used to cosmetically format plots
#

def format_plots():

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(
        font='serif',
        rc={
          'axes.axisbelow': False,
          'axes.edgecolor': 'lightgrey',
          'axes.facecolor': 'None',
          'axes.grid': False,
          'axes.labelcolor': 'dimgrey',
          'axes.spines.right': False,
          'axes.spines.top': False,
          'figure.facecolor': 'white',
          'lines.solid_capstyle': 'round',
          'patch.edgecolor': 'w',
          'patch.force_edgecolor': True,
          'text.color': 'black',
          'xtick.bottom': False,
          'xtick.color': 'dimgrey',
          'xtick.direction': 'out',
          'xtick.top': False,
          'ytick.color': 'dimgrey',
          'ytick.direction': 'out',
          'ytick.left': False,
          'ytick.right': False}
    )
    sns.set_context(
        "notebook", 
        rc={
          'font.size':14,
          'axes.titlesize':14,
          'axes.labelsize':14}
    )

    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#2CBDFE', '#47DBCD', '#F3A0F2', '#9D2EC5', '#661D98', '#F5B14C'
    ])
    sns.despine(left=True, bottom=True)

format_plots()
```


    <Figure size 720x576 with 0 Axes>



```python
#
# Read files from bucket
#

def bucket_to_bytes(file_name, bucket_name=BUCKET_NAME):

    from google.cloud import storage
    from io import BytesIO

    if f"gs://{bucket_name}/" in file_name:
        file_name = file_name.replace(f"gs://{bucket_name}/", "")

    # Get the blob
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Convert to DataFrame
    contents = blob.download_as_bytes()
    data = BytesIO(contents)
    return data
```


```python
#
# Read csv file as a Pandas DataFrame from the bucket
#

def read_csv_from_bucket(gcs_path, bucket_name=BUCKET_NAME):

    import pandas as pd
    csv_data = bucket_to_bytes(gcs_path, bucket_name)
    return pd.read_csv(csv_data)
```


```python
#
# Write csv file to the bucket
#

def write_csv_to_bucket(csv_object, gcs_path, bucket_name=BUCKET_NAME):

    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(gcs_path).upload_from_string(csv_object, 'text/csv')
```


```python
#
# Evaluation of Vertex AI classification model
#

def model_evaluation(model, cm_cols=['retain', 'churn'], highlight_feature=None):

    import pandas as pd
    import matplotlib.pyplot as plt

    for model_evaluation in model.list_model_evaluations():

        info_dict = model_evaluation.to_dict()
        info_metrics = info_dict['metrics']
        info_features = info_dict['modelExplanation']
        confusion_matrix = info_metrics['confusionMatrix']
        au_prc = info_metrics['auPrc']
        au_roc = info_metrics['auRoc']
        log_loss = info_metrics['logLoss']

    info_dict['cm_df'] = pd.DataFrame(confusion_matrix['rows'], 
        index=cm_cols, columns=cm_cols)
    TP = confusion_matrix['rows'][1][1]
    FP = confusion_matrix['rows'][0][1]
    FN = confusion_matrix['rows'][1][0]
    precision = round(TP / (FP + TP), 2)
    recall = round(TP / (FN + TP), 2)

    print("Model metrics:\n")
    print(f" - Area under Precision-Recall Curve: {au_prc}")
    print(f" - Area under Receiver Operating Characteristic Curve: {au_roc}")
    print(f" - Log-loss: {log_loss}")
    print(f" - Precision: {precision}")
    print(f" - Recall: {recall}")

    print("\nConfusion matrix:\n")
    print(info_dict['cm_df'])

    feature_importance = info_features['meanAttributions'][0]['featureAttributions']
    sorted_feature_importance = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=False)
    names = [feature for feature, _ in sorted_feature_importance]
    values = [value for _, value in sorted_feature_importance]

    plt.barh(range(len(feature_importance)), values, tick_label=names)
    if highlight_feature in names:
        plt.gca().get_yticklabels()[int(names.index(highlight_feature))].set_color("red")
    plt.title("Feature importance")
    plt.show()
    
    return info_dict
```


```python
#
# Optimise threshold classification by using costs
#

def optimisation_by_cost_matrix(tresholds, f_p_cost, f_n_cost, p_c):

    import pandas as pd
    import matplotlib.pyplot as plt

    costs = []
    for i in range(len(tresholds)):
      try:
        false_negative = int(tresholds[i]['falseNegativeCount'])
      except KeyError:
        false_negative = 0
      
      try:
        false_positive = int(tresholds[i]['falsePositiveCount'])
      except KeyError:
        false_positive = 0

      costs.append(f_p_cost*false_positive + f_n_cost*false_negative*p_c)

    best_threshold = tresholds[int(costs.index(min(costs)))]['confidenceThreshold']
    print(f"Minimum costs of {min(costs)}$ are obtained for {best_threshold} \
    threshold.")

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(np.linspace(0,1,len(tresholds)),costs)
    ax.scatter(best_threshold,min(costs), c='r', linewidths=25)
    ax.set(xlabel='threshold', ylabel='cost [$]',
          title='Cost VS threshold')
    ax.grid()

    plt.show()
    return best_threshold
```


```python
#
# Get predictions in batch mode
#

def get_df_from_batch_predict(
    input_file,
    output_folder,
    model_name,
    bucket_name,
    model_object=None,
    model_path=None,
):
    import pandas as pd
    
    if model_path is None:

        if model_object is None:
            raise AttributeError("If model_path is None, than model_object\
             need to be instance of model")
            
        print(f"Batch predictions with {model_name} model.")
        
        batch_predict_job = model_object.batch_predict(
          gcs_source=input_file,
          instances_format="csv",
          gcs_destination_prefix=output_folder,
          predictions_format="csv",
          job_display_name=f"job-batch_predict-{model_name}-{TIMESTAMP}",
          sync=True
        )
        batch_predict_job.wait()
        list_files = batch_predict_job.iter_outputs()
        print(batch_predict_job.output_info.gcs_output_directory)

    else:
        
        from google.cloud import storage
        
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        list_files = list(bucket.list_blobs(prefix=model_path))

    df_list = []
    for row in list_files:
        df_list.append(read_csv_from_bucket(
            row.name, bucket_name=bucket_name))

    return pd.concat(df_list)
```


```python
#
# Plot time-series of prediction churn periods 
#

def plot_uuid_ts_with_periods(
    uuid, 
    df_predict_interaction, 
    df_lt_result, 
    thresholds,
    map_int_to_date):
    
    import matplotlib.pyplot as plt
    import pandas as pd
    from dateutil.relativedelta import relativedelta
  
    print(f"User id: {uuid}")

    fig_size = (10, 15)
    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=fig_size, 
                            gridspec_kw={'height_ratios': [1, 1, 1, 2]})
    for i, col in enumerate(['internet_min', 'phone_min', 'number_customer_service_calls']):

        df_tmp = df_predict_interaction.loc[(df_predict_interaction['uuid']==uuid)].iloc[-7:]

        df_tmp.index = pd.to_datetime(df_tmp['timestamp'])
        df_tmp_pred = df_lt_result.loc[(df_lt_result['uuid']==uuid)]
        df_tmp_pred.index = pd.to_datetime(df_tmp_pred['timestamp'])

        df_tmp_pred = df_lt_result.loc[(df_lt_result['timestamp']=='2021-10-01')&(df_lt_result['uuid']==uuid)
          ].sort_values(by='time_horizon')[['time_horizon', 'churn', 'churn_1_scores', col]]
        df_tmp_pred = df_tmp_pred.replace({"time_horizon": map_int_to_date})
        df_tmp_pred.index = pd.to_datetime(df_tmp_pred['time_horizon'])

        df_tmp.loc[df_tmp['uuid']==uuid, [col]].plot(ax=axs[i])
        x_true_neg = df_tmp_pred.loc[(df_tmp_pred['churn']==0)].index
        x_true_pos = df_tmp_pred.loc[(df_tmp_pred['churn']==1)].index


        # x_pred_neg = df_tmp_pred.loc[(df_tmp_pred['churn_1_scores']<=.5)].index
        # x_pred_pos = df_tmp_pred.loc[(df_tmp_pred['churn_1_scores']>=.5)].index

        # max_int = max(df_tmp[col].max(), 1)

        if len(x_true_pos)>0:
            x_true_pos = sorted([min(list(x_true_pos)) - relativedelta(months=1)] + list(x_true_pos))
            x_true_pos = [list(x_true_pos)[0] + relativedelta(months=1)] + list(x_true_pos)
            y = df_tmp.loc[df_tmp.index.isin(x_true_pos), col].values
            if len(y)< len(x_true_pos):
                x_true_pos = x_true_pos[1:]
            axs[i].fill_between(x_true_pos, y, color='red', alpha=.2, label='true churn period')

        if len(x_true_neg)>0:
            x_true_neg = sorted([min(list(x_true_neg)) - relativedelta(months=1)] + list(x_true_neg))
            
            y = df_tmp.loc[df_tmp.index.isin(x_true_neg), col].values
            if len(y)< len(x_true_neg):
                x_true_neg = x_true_neg[1:]
            axs[i].fill_between(x_true_neg, y, color='blue', alpha=.2, label='true retention period')
        axs[i].legend(loc='upper left')
        axs[i].set_title(f"True values for \nperiods of {col} ts feature")
        axs[i].set_xlabel('Months')
        axs[i].set_ylabel('min' if col != 'no_service_calls' else 'count')
    
    if len(df_tmp_pred.index) >= len(map_int_to_date):
        axs[3].bar(
            df_tmp_pred.index, 
            df_tmp_pred['churn_1_scores'].values,
            width = 10,
            label = 'churn probability',
            )
        axs[3].step(
            df_tmp_pred.index.values, 
            list(thresholds.values()), 
            'k',
            where="mid",
            linestyle='--',
            linewidth=2,
            label = 'threshold'
            )

    axs[3].set_title(f"Risk of churning")
    axs[3].legend(loc='center left')
    plt.show()
```

## Load data

In many cases, user data is very sensitive and because of it's complience it cannot be used as it is for the purpose of this notbook. Moreover, if we took into consideration that main purpose of this starting kit is to show best practices for implementing churn prediction methodology with Vertex AI, it is not crucial to have real data for demonstration purposes and therefore we decided to proceed with syntheticly generated data. There are three types of data generated for this exercise, and that is:

 - Account data - this is data that is usually stored in data warehouses and it consist from user social-demografic, duration of contract, type of payment and etc.
 - Interaction data - this is data that is usually stored in transactional database and it consisted from user generated events, such as payment transactions, usage of system during time, logs and etc. 
 - User generated data - this data is usually stored in data lake and it consisted of unstructural data, such as email, call transcription, chats and etc.

### Account data


```python
#
# Read account data
#

CSV_TRAIN_ACCOUNT = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_short_term_train'])
CSV_PREDICT_ACCOUNT = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_short_term_predict'])
DATASET_ACCOUNT = aiplatform.TabularDataset.create(
    display_name="account_data_train", gcs_source=[CSV_TRAIN_ACCOUNT])
```

    Creating TabularDataset


    INFO:google.cloud.aiplatform.datasets.dataset:Creating TabularDataset


    Create TabularDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/8054602171645165568/operations/2518503195274117120


    INFO:google.cloud.aiplatform.datasets.dataset:Create TabularDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/8054602171645165568/operations/2518503195274117120


    TabularDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/8054602171645165568


    INFO:google.cloud.aiplatform.datasets.dataset:TabularDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/8054602171645165568


    To use this TabularDataset in another session:


    INFO:google.cloud.aiplatform.datasets.dataset:To use this TabularDataset in another session:


    ds = aiplatform.TabularDataset('projects/1099418293219/locations/us-central1/datasets/8054602171645165568')


    INFO:google.cloud.aiplatform.datasets.dataset:ds = aiplatform.TabularDataset('projects/1099418293219/locations/us-central1/datasets/8054602171645165568')



```python
df_train_account = read_csv_from_bucket(CSV_TRAIN_ACCOUNT)
df_predict_account = read_csv_from_bucket(CSV_PREDICT_ACCOUNT)
df_train_account.head()
```





  <div id="df-151f4dc2-f7d9-4966-90af-153dfebfce55">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uuid</th>
      <th>gender</th>
      <th>tenure</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>contract_duration</th>
      <th>payment_method</th>
      <th>number_customer_service_calls</th>
      <th>phone_min</th>
      <th>internet_min</th>
      <th>phone_monthly_charges</th>
      <th>internet_monthly_charges</th>
      <th>avg_monthly_bill</th>
      <th>churn</th>
      <th>trigger_price</th>
      <th>trigger_quality</th>
      <th>trigger_external</th>
      <th>treatment</th>
      <th>churn_after_treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9549ab1e-febf-11ec-ac44-0242ac1c0002</td>
      <td>Male</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>long-term</td>
      <td>automatic transfer</td>
      <td>3</td>
      <td>140.769699</td>
      <td>0.000000</td>
      <td>23.886935</td>
      <td>0.000000</td>
      <td>26.049606</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>954be69a-febf-11ec-ac44-0242ac1c0002</td>
      <td>Female</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>long-term</td>
      <td>mail</td>
      <td>5</td>
      <td>126.936760</td>
      <td>0.000000</td>
      <td>21.580243</td>
      <td>0.000000</td>
      <td>17.677925</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>upgrade_package</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>954e6b4a-febf-11ec-ac44-0242ac1c0002</td>
      <td>Male</td>
      <td>47</td>
      <td>1</td>
      <td>1</td>
      <td>long-term</td>
      <td>mail</td>
      <td>0</td>
      <td>102.840743</td>
      <td>4067.393031</td>
      <td>17.481386</td>
      <td>42.920395</td>
      <td>48.838830</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>955afd38-febf-11ec-ac44-0242ac1c0002</td>
      <td>Female</td>
      <td>54</td>
      <td>1</td>
      <td>1</td>
      <td>long-term</td>
      <td>email</td>
      <td>3</td>
      <td>121.610911</td>
      <td>13148.296507</td>
      <td>20.674387</td>
      <td>90.895169</td>
      <td>106.498240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>954a73e6-febf-11ec-ac44-0242ac1c0002</td>
      <td>Male</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>3</td>
      <td>106.279190</td>
      <td>6448.168722</td>
      <td>18.067917</td>
      <td>39.407477</td>
      <td>58.549109</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-151f4dc2-f7d9-4966-90af-153dfebfce55')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-151f4dc2-f7d9-4966-90af-153dfebfce55 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-151f4dc2-f7d9-4966-90af-153dfebfce55');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




*   `uuid` - unique user id
*   `gender` - Male/Female
*   `tenure` - number of consecutive months that user is subscribed to a service
*   `phone_services` - 0 or 1 that shows if user use phone services
*   `internet_services` - 0 or 1 that shows if user use internet services
*   `contract_duration` - short/long type of contract
*   `payment_method` - email/mail/automatic type of payment
*   `number_customer_service_calls` - total number of calls to the customer services
*   `phone_min` - total duration of phone calls in last month
*   `internet_min` - total duration of internet usage in last month
*   `phone_monthly_charges` - phone call bill for last month
*   `internet_monthly_charges` - internet bill for last month
*   `avg_monthly_bill` - average bill during user lifetime
*   `trigger_price` - 0 or 1 that shows if price trigger user to churn
*   `trigger_quality` - 0 or 1 that shows if quality of service trigger user to churn
*   `trigger_external` - 0 or 1 that shows if external factor trigger user to churn
*   `churn` - 0 or 1 that shows if user churn or not
*   `treatment` - none/discount/upgrade package/free device treatment implemented in order to retain user
*   `churn_after_treatment` - 0 or 1 that shows if user retain after treatment

Usually, churn datasets are imbalanced, that means that there are more retaine users than churned ones. Consequently, this is the case in current dataset as well, with 26% of churned users. Moreover, this will be one of the point that we need to address during the training phase.


```python
ax = df_train_account.replace(
        {'churn': {0: "Retain users", 1: "Churn users"}}
    ).groupby('churn')['churn'].count().plot(
        kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], 
        legend=True, title='Ratio between churn and retain users', 
        ylabel='', labeldistance=None
    )
    
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 15})

plt.show()
```


    
![png](output_30_0.png)
    


### Interaction data


```python
CSV_TRAIN_INTERACTION = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_long_term_v1_train'])
CSV_PREDICT_INTERACTION = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_long_term_v1_predict'])
DATASET_INTERACTION = aiplatform.TimeSeriesDataset.create(
  display_name="timeseries_data_train", gcs_source=[CSV_TRAIN_INTERACTION])
```

    Creating TimeSeriesDataset


    INFO:google.cloud.aiplatform.datasets.dataset:Creating TimeSeriesDataset


    Create TimeSeriesDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/2706577614142701568/operations/6633104404830486528


    INFO:google.cloud.aiplatform.datasets.dataset:Create TimeSeriesDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/2706577614142701568/operations/6633104404830486528


    TimeSeriesDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/2706577614142701568


    INFO:google.cloud.aiplatform.datasets.dataset:TimeSeriesDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/2706577614142701568


    To use this TimeSeriesDataset in another session:


    INFO:google.cloud.aiplatform.datasets.dataset:To use this TimeSeriesDataset in another session:


    ds = aiplatform.TimeSeriesDataset('projects/1099418293219/locations/us-central1/datasets/2706577614142701568')


    INFO:google.cloud.aiplatform.datasets.dataset:ds = aiplatform.TimeSeriesDataset('projects/1099418293219/locations/us-central1/datasets/2706577614142701568')



```python
df_train_interaction = read_csv_from_bucket(CSV_TRAIN_INTERACTION)
df_predict_interaction = read_csv_from_bucket(CSV_PREDICT_INTERACTION)
df_train_interaction.head()
```





  <div id="df-7edd5f67-316e-4d45-af12-7621177a1c68">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>phone_min</th>
      <th>internet_min</th>
      <th>number_customer_service_calls</th>
      <th>uuid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-03-01</td>
      <td>122.558023</td>
      <td>5015.185948</td>
      <td>0.0</td>
      <td>8a40951c-0756-11ed-a65f-0242ac1c0002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-01</td>
      <td>108.878533</td>
      <td>4670.185744</td>
      <td>0.0</td>
      <td>8a4095e4-0756-11ed-a65f-0242ac1c0002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-03-01</td>
      <td>0.000000</td>
      <td>1850.548032</td>
      <td>0.0</td>
      <td>8a409698-0756-11ed-a65f-0242ac1c0002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-03-01</td>
      <td>0.000000</td>
      <td>2853.754850</td>
      <td>0.0</td>
      <td>8a4097b0-0756-11ed-a65f-0242ac1c0002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-03-01</td>
      <td>102.987957</td>
      <td>4287.638431</td>
      <td>0.0</td>
      <td>8a409828-0756-11ed-a65f-0242ac1c0002</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7edd5f67-316e-4d45-af12-7621177a1c68')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7edd5f67-316e-4d45-af12-7621177a1c68 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7edd5f67-316e-4d45-af12-7621177a1c68');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




*   `uuid` - unique user id
*   `timestamp` - time when record is generated
*   `phone_min` - total duration of phone calls during time window of one month
*   `internet_min` - total duration of internet usage during time window of one month
*   `number_customer_service_calls` - total number of calls to the customer services during time window of one month


### Combine data-set

In this part we are loading data that obtained by merging accoutn and interaction data, combined in a way so that could be used for training Tabular models. That is, we introduced several new features like internet_min_lag1, internet_min_lag2, internet_min_lag3 and etc. where we add data from previous period, while we add a feature time_horizont for determine for which period in a future we want to make a prediction.


```python
CSV_TRAIN_ACC_INTER = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_combined_data_train'])
CSV_PREDICT_ACC_INTER = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_combined_data_predict'])
DATASET_ACC_INTER = aiplatform.TabularDataset.create(
    display_name="account_data_train", gcs_source=[CSV_TRAIN_ACC_INTER])
```

    Creating TabularDataset


    INFO:google.cloud.aiplatform.datasets.dataset:Creating TabularDataset


    Create TabularDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/4759093144316805120/operations/4648705819020361728


    INFO:google.cloud.aiplatform.datasets.dataset:Create TabularDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/4759093144316805120/operations/4648705819020361728


    TabularDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/4759093144316805120


    INFO:google.cloud.aiplatform.datasets.dataset:TabularDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/4759093144316805120


    To use this TabularDataset in another session:


    INFO:google.cloud.aiplatform.datasets.dataset:To use this TabularDataset in another session:


    ds = aiplatform.TabularDataset('projects/1099418293219/locations/us-central1/datasets/4759093144316805120')


    INFO:google.cloud.aiplatform.datasets.dataset:ds = aiplatform.TabularDataset('projects/1099418293219/locations/us-central1/datasets/4759093144316805120')



```python
df_train_acc_inter = read_csv_from_bucket(CSV_TRAIN_ACC_INTER)
df_predict_acc_inter = read_csv_from_bucket(CSV_PREDICT_ACC_INTER)
df_train_acc_inter.head()
```





  <div id="df-b96c3903-a7ca-42eb-8c70-c27444dcc736">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uuid</th>
      <th>gender</th>
      <th>tenure</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>contract_duration</th>
      <th>payment_method</th>
      <th>phone_monthly_charges</th>
      <th>internet_monthly_charges</th>
      <th>avg_monthly_bill</th>
      <th>...</th>
      <th>internet_min_lag_4</th>
      <th>internet_min_lag_5</th>
      <th>internet_min_lag_6</th>
      <th>number_customer_service_calls_lag_1</th>
      <th>number_customer_service_calls_lag_2</th>
      <th>number_customer_service_calls_lag_3</th>
      <th>number_customer_service_calls_lag_4</th>
      <th>number_customer_service_calls_lag_5</th>
      <th>number_customer_service_calls_lag_6</th>
      <th>time_horizon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a409828-0756-11ed-a65f-0242ac1c0002</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>short-term</td>
      <td>mail</td>
      <td>17.510440</td>
      <td>30.371958</td>
      <td>25.025</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a409a94-0756-11ed-a65f-0242ac1c0002</td>
      <td>Male</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>short-term</td>
      <td>mail</td>
      <td>21.169812</td>
      <td>32.099249</td>
      <td>27.088</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a409b0c-0756-11ed-a65f-0242ac1c0002</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>short-term</td>
      <td>email</td>
      <td>19.680637</td>
      <td>28.012835</td>
      <td>23.160</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a40a912-0756-11ed-a65f-0242ac1c0002</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>mail</td>
      <td>0.000000</td>
      <td>5.509705</td>
      <td>14.185</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a40ab60-0756-11ed-a65f-0242ac1c0002</td>
      <td>Male</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>short-term</td>
      <td>mail</td>
      <td>18.361349</td>
      <td>31.365313</td>
      <td>25.351</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b96c3903-a7ca-42eb-8c70-c27444dcc736')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b96c3903-a7ca-42eb-8c70-c27444dcc736 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b96c3903-a7ca-42eb-8c70-c27444dcc736');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
CSV_TRAIN_ACC_INTER_V2 = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_long_term_v2_train'])
CSV_PREDICT_ACC_INTER_V2 = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_long_term_v2_predict'])
DATASET_ACC_INTER_V2 = aiplatform.TimeSeriesDataset.create(
  display_name="timeseries_data_train_forecast", gcs_source=[CSV_TRAIN_ACC_INTER_V2])
```

    Creating TimeSeriesDataset


    INFO:google.cloud.aiplatform.datasets.dataset:Creating TimeSeriesDataset


    Create TimeSeriesDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/2591735823644753920/operations/6393287724673007616


    INFO:google.cloud.aiplatform.datasets.dataset:Create TimeSeriesDataset backing LRO: projects/1099418293219/locations/us-central1/datasets/2591735823644753920/operations/6393287724673007616


    TimeSeriesDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/2591735823644753920


    INFO:google.cloud.aiplatform.datasets.dataset:TimeSeriesDataset created. Resource name: projects/1099418293219/locations/us-central1/datasets/2591735823644753920


    To use this TimeSeriesDataset in another session:


    INFO:google.cloud.aiplatform.datasets.dataset:To use this TimeSeriesDataset in another session:


    ds = aiplatform.TimeSeriesDataset('projects/1099418293219/locations/us-central1/datasets/2591735823644753920')


    INFO:google.cloud.aiplatform.datasets.dataset:ds = aiplatform.TimeSeriesDataset('projects/1099418293219/locations/us-central1/datasets/2591735823644753920')



```python
df_train_acc_inter_v2 = read_csv_from_bucket(CSV_TRAIN_ACC_INTER_V2)
df_predict_acc_inter_v2 = read_csv_from_bucket(CSV_PREDICT_ACC_INTER_V2)
df_train_acc_inter_v2.head()
```





  <div id="df-02911425-995a-4af0-8bcf-0607e0dd2b97">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uuid</th>
      <th>timestamp</th>
      <th>gender</th>
      <th>phone_services</th>
      <th>internet_services</th>
      <th>contract_duration</th>
      <th>payment_method</th>
      <th>avg_monthly_bill</th>
      <th>phone_duration</th>
      <th>internet_duration</th>
      <th>no_service_calls</th>
      <th>churn_regr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f3d4102a-1a17-11ed-8483-0242ac1c0002</td>
      <td>2020-09-01</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>34.655143</td>
      <td>0.0</td>
      <td>2607.506549</td>
      <td>0.0</td>
      <td>-23.025851</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f3d4102a-1a17-11ed-8483-0242ac1c0002</td>
      <td>2020-10-01</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>34.655143</td>
      <td>0.0</td>
      <td>2797.806073</td>
      <td>0.0</td>
      <td>-23.025851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f3d4102a-1a17-11ed-8483-0242ac1c0002</td>
      <td>2020-11-01</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>34.655143</td>
      <td>0.0</td>
      <td>2502.989636</td>
      <td>0.0</td>
      <td>-23.025851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f3d4102a-1a17-11ed-8483-0242ac1c0002</td>
      <td>2020-12-01</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>34.655143</td>
      <td>0.0</td>
      <td>2464.037121</td>
      <td>0.0</td>
      <td>-23.025851</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f3d4102a-1a17-11ed-8483-0242ac1c0002</td>
      <td>2021-01-01</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>short-term</td>
      <td>automatic transfer</td>
      <td>34.655143</td>
      <td>0.0</td>
      <td>2605.847246</td>
      <td>0.0</td>
      <td>-23.025851</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-02911425-995a-4af0-8bcf-0607e0dd2b97')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-02911425-995a-4af0-8bcf-0607e0dd2b97 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-02911425-995a-4af0-8bcf-0607e0dd2b97');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Short-term churn prediction

### Train model

As it was mantion previously, dataset used for training this model is imbalanced. That means, that more users retain in the system than the ones that churnes. Therefor, we are going to use 'maximize-au-prc' objective function. By looking at Vertex AI documentation, we may see that this function is used in a case when we want to optimize for less common class (in this case churned users), and it stands for area under the precision-recall curve.


```python
exclude = ['uuid']
target_column = 'churn'
multi_target_columns = ['trigger_price', 'trigger_quality', 'trigger_external']
treatments = ['churn_after_treatment', 'treatment']
cols = [col for col in df_train_account.columns 
        if col not in exclude+[target_column]+multi_target_columns+treatments]
categorical_cols = df_train_account.loc[:, cols].select_dtypes(exclude="number").columns
numerical_cols = df_train_account.loc[:, cols].select_dtypes(include="number").columns

COLUMN_SPEC = {}
for col_i in numerical_cols:
  COLUMN_SPEC[col_i] = "numeric"
for col_i in categorical_cols:
  COLUMN_SPEC[col_i] = "categorical"
```


```python
model_st_id = config['artifacts']['model_id']['short_term_churn_model']
model_st_display_name = "aggregate-churn-prediction-model"

if model_st_id is None:
  job_tabular = aiplatform.AutoMLTabularTrainingJob(
      display_name=f'job_train_model-{model_st_display_name}-{TIMESTAMP}',
      optimization_prediction_type='classification',
      optimization_objective='maximize-au-prc',
      column_specs=COLUMN_SPEC
  )

  model_st = job_tabular.run(
      dataset = DATASET_ACCOUNT,
      target_column = target_column,
      training_fraction_split = 0.7,
      validation_fraction_split = 0.15,
      test_fraction_split = 0.15,
      budget_milli_node_hours=10, #1000
      model_display_name=model_st_display_name,
  )
else:
  model_st = aiplatform.Model(model_st_id)
```

### Analyse model


```python
info_eval = model_evaluation(model_st)
```

    Model metrics:
    
     - Area under Precision-Recall Curve: 0.9908749
     - Area under Receiver Operating Characteristic Curve: 0.9905493
     - Log-loss: 0.12698692
     - Precision: 0.9
     - Recall: 0.87
    
    Confusion matrix:
    
            retain  churn
    retain   822.0   28.0
    churn     38.0  256.0



    
![png](output_47_1.png)
    


Feature importance reveals which features contributes the most in classifing churners. Therefore, we are going to utilise it to take a closer inspection of distributions of the most promenent ones (average monthly bills, charges for internet and internet minutes). In addition, comparing to other features, this graph shows that gender, contract duration, phone and internet services plays little role in defining churners.


```python
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 10))
axs[0].set_title("Compare features of churns and retain users")
axs[0].hist(
    [df_train_account.loc[df_train_account['churn']==1, 'internet_monthly_charges'].tolist(),
     df_train_account.loc[df_train_account['churn']==0, 'internet_monthly_charges'].tolist()], 
     label=['churn','retain'],
     density=True)
axs[0].set_xlabel("internet monthly charges [$]")
axs[0].legend(loc='upper right')
axs[1].hist(
    [df_train_account.loc[df_train_account['churn']==1, 'internet_min'].tolist(),
     df_train_account.loc[df_train_account['churn']==0, 'internet_min'].tolist()], 
     label=['churn','retain'],
     density=True)
axs[1].set_xlabel("internet usage [min]")
axs[2].hist(
    [df_train_account.loc[df_train_account['churn']==1, 'tenure'].tolist(),
     df_train_account.loc[df_train_account['churn']==0, 'tenure'].tolist()], 
     label=['churn','retain'],
     density=True)
axs[2].set_xlabel("user tenure [months]")
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/numpy/core/fromnumeric.py:3208: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return asarray(a).size
    /usr/local/lib/python3.7/dist-packages/matplotlib/cbook/__init__.py:1376: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      X = np.atleast_1d(X.T if isinstance(X, np.ndarray) else np.asarray(X))



    
![png](output_49_1.png)
    


This type of analyses is out of the scope of this notebook, but it could be implemented further on in defining churn metrics. This type of metrics is valuable in buisness as a measure of showing how users that retain in a system use their product versus the ones that churned. Moreover, it could be used to push churners through customer support or some treatment which will convert them back to the services.

### Optimisation by cost matrix

In general, in order to have better precision we need to sacrifice recall, and vice versa. Moreover, different businesses has different sensitivity when it comes to selecting recall vs precision. So, defining balance between those two metrics represents an important decision, because it will revield what threshold we should use and how it will effect our costs. One of the best practices in this case is to use cost matrix. 

In order to illustrate this, we add artificially costs as follow:
- Recall
 - costs of replacing false pnegative users with new ones cost on average 40 dollars, 
 - probability of keeping false negative if we would clasiffy them as churners is 0.3
 - Therefore, expected costs of not predicting churned users per case is 12 dollars. 
- Precision
 - Wrongly treated retain users as churned ones cost company 10 dollars per case, by promoting the system through one time down sell promotions.


```python
import numpy as np
import pandas as pd


# illustrative values
down_sell_cost = 10
acquisition_cost = 40
prob_of_converting = 0.3


expected_return_TN = df_train_account.loc[df_train_account.churn==0].avg_monthly_bill.mean()
expected_return_TP = prob_of_converting * (
    df_train_account.loc[df_train_account.churn==1].avg_monthly_bill.mean() - 
    down_sell_cost)
expected_cost_FP = -1* down_sell_cost
expected_cost_FN = -1* (acquisition_cost - down_sell_cost) * prob_of_converting
cost_matrix = pd.DataFrame(
    np.array([
                        [expected_return_TN, expected_cost_FP], 
                        [expected_cost_FN, expected_return_TP]
]).astype(int), index=['retain', 'churn'], columns=['retain', 'churn'])
cost_matrix.style.apply(
    lambda x: ['background-color: red' if v<=0 else "" for v in x]
)
```




<style type="text/css">
#T_d13e8_row0_col1, #T_d13e8_row1_col0 {
  background-color: red;
}
</style>
<table id="T_d13e8_" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >retain</th>
      <th class="col_heading level0 col1" >churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d13e8_level0_row0" class="row_heading level0 row0" >retain</th>
      <td id="T_d13e8_row0_col0" class="data row0 col0" >57</td>
      <td id="T_d13e8_row0_col1" class="data row0 col1" >-10</td>
    </tr>
    <tr>
      <th id="T_d13e8_level0_row1" class="row_heading level0 row1" >churn</th>
      <td id="T_d13e8_row1_col0" class="data row1 col0" >-9</td>
      <td id="T_d13e8_row1_col1" class="data row1 col1" >15</td>
    </tr>
  </tbody>
</table>





```python
best_threshold = optimisation_by_cost_matrix(
    info_eval['metrics']['confidenceMetrics'], 
    down_sell_cost, 
    acquisition_cost, 
    prob_of_converting)
```

    Minimum costs of 1306.0$ are obtained for 0.42     threshold.



    
![png](output_54_1.png)
    


## Long-term churn prediction

## Version 1: AutoMLTabular model

This section of the notebook address the problem of identifying churners before it is too late. Therefore, we will show the model that is capable of providing estimations several months before users unsubscribe from services. For this purpose, we implemented synthetically generated time series data, that is used for training time aware model with autoML Tabular (Vertex AI) service.

Furthermore, we show that short-term models are better balanced and that they are more sensitive to churn behaviour. However, major downside of short-term models is that after it make predictions, buisness doesn't have so much time to act and in many cases users already made definite decision of churning from product. However, this is where long term predictions shine and it gave us ability to act on a time by predicting probability of churning for mid future period (6 months in this case). 

In general, we would like to keep both models, short-term models because they are more accurate and it reveals actual user intentions, while long-term model is better with identification of possible future churners.

### Train model


```python
exclude = ['uuid', 'timestamp']
target_column = 'churn'
multi_target_columns = ['trigger_price', 'trigger_quality', 'trigger_external']
treatments = ['churn_after_treatment', 'treatment']
cols = [col for col in df_train_acc_inter.columns 
        if col not in exclude+[target_column]+multi_target_columns+treatments]
categorical_cols = df_train_acc_inter.loc[:, cols].select_dtypes(exclude="number").columns
numerical_cols = df_train_acc_inter.loc[:, cols].select_dtypes(include="number").columns

COLUMN_SPEC = {}
for col_i in numerical_cols:
  COLUMN_SPEC[col_i] = "numeric"
for col_i in categorical_cols:
  COLUMN_SPEC[col_i] = "categorical"
```


```python
model_lt_id = config['artifacts']['model_id']['long_term_churn_model_v1']
model_lt_display_name = "event-churn-prediction-model"

if model_lt_id is None:
  job_tabular = aiplatform.AutoMLTabularTrainingJob(
      display_name=f'job_train_model-{model_lt_display_name}-{TIMESTAMP}',
      optimization_prediction_type='classification',
      optimization_objective='maximize-au-prc',
      column_specs=COLUMN_SPEC
  )

  model_lt = job_tabular.run(
      dataset = DATASET_ACC_INTER,
      target_column = target_column,
      training_fraction_split = 0.7,
      validation_fraction_split = 0.15,
      test_fraction_split = 0.15,
      budget_milli_node_hours=10,
      model_display_name=model_lt_display_name,
  )
else:
  model_lt = aiplatform.Model(model_lt_id)
```

### Analyse model


```python
info_eval = model_evaluation(model_lt, highlight_feature='time_horizon')
```

    Model metrics:
    
     - Area under Precision-Recall Curve: 0.9735879
     - Area under Receiver Operating Characteristic Curve: 0.97260904
     - Log-loss: 0.22249362
     - Precision: 0.74
     - Recall: 0.53
    
    Confusion matrix:
    
            retain  churn
    retain  7111.0  240.0
    churn    615.0  685.0



    
![png](output_62_1.png)
    



```python
DATA_LT_FOLDER = config['artifacts']['model_path']['long_term_churn_model_v1_folder']
model_lt_results = config['data']['output_data']['file_path_long_term_churn_model_v1']
gsc_output_folder = os.path.join("gs://", BUCKET_NAME, DATA_LT_FOLDER, f"predictions-st-{TIMESTAMP}")

df_lt_result = get_df_from_batch_predict(
    CSV_PREDICT_ACC_INTER, 
    gsc_output_folder, 
    model_lt_display_name,
    BUCKET_NAME,
    model_lt,
    model_lt_results)

df_lt_result['predicted_churn'] = 0
df_lt_result.loc[
    df_lt_result['churn_0_scores']<=.5, 'predicted_churn'] = 1
```


```python
threshodls_by_time_horizont = {}
tresholds = np.linspace(0,1, 101)
for time_horizont in range(1,7):
    df_tmp = df_lt_result.loc[df_lt_result['time_horizon']==time_horizont, ['churn_1_scores', 'churn']]
    costs = []
    for th in tresholds:
        FP = df_tmp.loc[(df_tmp['churn']==0)& (df_tmp['churn_1_scores']>=th)].shape[0]
        FN = df_tmp.loc[(df_tmp['churn']==1)& (df_tmp['churn_1_scores']<th)].shape[0]
        costs.append(
            down_sell_cost*FP + 
            acquisition_cost*FN*prob_of_converting
        )
    threshodls_by_time_horizont[time_horizont] = tresholds[int(costs.index(min(costs)))]
```


```python
map_dates = {
    1: '2021-10-01',
    2: '2021-11-01',
    3: '2021-12-01',
    4: '2022-01-01',
    5: '2022-02-01',
    6: '2022-03-01'
}
```


```python
plot_uuid_ts_with_periods(
        '8a43ef1e-0756-11ed-a65f-0242ac1c0002', 
        df_predict_interaction, 
        df_lt_result, 
        threshodls_by_time_horizont,
        map_dates
    )
```

    User id: 8a43ef1e-0756-11ed-a65f-0242ac1c0002



    
![png](output_66_1.png)
    



```python
plot_uuid_ts_with_periods(
        '8a4721ca-0756-11ed-a65f-0242ac1c0002', 
        df_predict_interaction, 
        df_lt_result, 
        threshodls_by_time_horizont,
        map_dates
    )
```

    User id: 8a4721ca-0756-11ed-a65f-0242ac1c0002



    
![png](output_67_1.png)
    


## Version 2: AutoMLForecasting model

In implementing sequence model, we faced with two major challenges. 

1.   One of them were how to transform classification to regression problem. Basically, churn prediction labels in data represent revealed probability that user will churn, therefore we have only two labels 0 that user retain and 1 that user churn. By interpreting a label as probability of churnes, the problem is reframed as a regression one. Therefore, we may use inverse sigmoid function to transform labels in a more easilly "digested" values for time serias model. 

> $$ x_{regression} =  \dfrac{1}{1+e^{-x}} $$

2.   Second, dataset is even more unbalanced, that is churnes represent rare events in time serias. We approach this problem, by smoothing values near churn event with 0.5 probability values that represent uninformative information, and underline justification is that we are uncertanty whether user decided to churn during that moment in time. Moreover, we trim long historical non churn sequence, in order to reduce non churn events.



### Train model


```python
target_column = 'churn_regr'
time_column = 'timestamp'
time_series_identifier_column = 'uuid'
available_at_forecast_columns = [
  'timestamp', 'phone_duration',	'internet_duration',	
  'no_service_calls']
time_series_attribute_columns = [
  'gender',	'phone_services',	'internet_services',	
  'contract_duration',	'payment_method',	'avg_monthly_bill']
cols = [col for col in df_train_acc_inter_v2.columns if col not in 
        [time_column, time_series_identifier_column]]
categorical_cols = df_train_acc_inter_v2.loc[:, cols].select_dtypes(exclude="number").columns
numerical_cols = df_train_acc_inter_v2.loc[:, cols].select_dtypes(include="number").columns

COLUMN_SPEC = {
    time_column: 'timestamp',
    target_column: 'churn_regr'
}
for col_i in numerical_cols:
  COLUMN_SPEC[col_i] = "numeric"
for col_i in categorical_cols:
  COLUMN_SPEC[col_i] = "categorical"

forecast_horizon = 3
context_window = 6
model_ts_display_name = "timeseries-churn-prediction-model-regr"
```


```python
model_ts_id = config['artifacts']['model_id']['long_term_churn_model_v2']
if model_ts_id is None:
  job_long_term = aiplatform.AutoMLForecastingTrainingJob(
      display_name=f'job_train_model-{model_ts_display_name}-{TIMESTAMP}',
      optimization_objective='minimize-rmse', #'minimize-rmse',#'minimize-quantile-loss', #'minimize-mae',    
      column_specs = COLUMN_SPEC,
  )

  # This will take around an hour to run
  model_ts = job_long_term.run(
      dataset=DATASET_ACC_INTER_V2,
      target_column=target_column,
      time_column=time_column,
      time_series_identifier_column=time_series_identifier_column,
      available_at_forecast_columns=available_at_forecast_columns,
      unavailable_at_forecast_columns=[target_column],
      time_series_attribute_columns=time_series_attribute_columns,
      forecast_horizon=forecast_horizon,
      context_window=context_window,
      data_granularity_unit="month",
      data_granularity_count=1,
      weight_column=None,
      budget_milli_node_hours=1000,
      model_display_name=model_ts_display_name, 
      predefined_split_column_name=None
  )
else:
  model_ts = aiplatform.Model(model_ts_id)
```


```python
print("Model evaluation performance: ")
for model_ts_evaluation in model_ts.list_model_evaluations():
  info_dict = model_ts_evaluation.to_dict()
  for metric, value in info_dict['metrics'].items():
    print(f"  - Model {metric}: {value}")
```

    Model evaluation performance: 
      - Model rootMeanSquaredLogError: 0.7448453
      - Model rootMeanSquaredError: 9.569928
      - Model rSquared: 0.99662787
      - Model meanAbsoluteError: 5.2578797
      - Model meanAbsolutePercentageError: 22.83468



```python
DATA_ACC_INTER_FOLDER_V2 = config['artifacts']['model_path']['long_term_churn_model_v2_folder']
model_lt_results = config['data']['output_data']['file_path_long_term_churn_model_v2']
gsc_output_folder = os.path.join("gs://", BUCKET_NAME, DATA_ACC_INTER_FOLDER_V2, f"predictions-{TIMESTAMP}")

df_ts_result = get_df_from_batch_predict(
    CSV_PREDICT_ACC_INTER_V2, 
    gsc_output_folder, 
    model_ts_display_name,
    BUCKET_NAME,
    model_ts,
    model_lt_results)

# merge true label
df_ts_result = df_ts_result.merge(
    df_predict_acc_inter_v2[['timestamp', 'uuid', 'true_label']], 
    on=['timestamp', 'uuid'])

# convert label and predict to prob values from 0 to 1
df_ts_result['predicted_churn'] = 1/(1+np.exp(-df_ts_result['predicted_churn_regr']))
df_ts_result['true_label'] = 1/(1+np.exp(-df_ts_result['true_label']))

df_ts_result.sort_values(by=['uuid', 'timestamp'], inplace=True)
```

## Compare Sequence (V2) with Tabular (V1) approach


```python
cm_sequence_model = pd.crosstab(
    df_ts_result['true_label'].round(0), 
    df_ts_result['predicted_churn'].round(0), 
    rownames=['Actual'], 
    colnames=['Predicted'], 
    margins=False
    ).rename({0: 'retain', 1:'churn'}
    ).rename({0: 'retain', 1:'churn'}, axis=1)

precision_forecaster = round(
    cm_sequence_model.loc['churn', 'churn'] / 
    cm_sequence_model.loc[:, 'churn'].sum(), 2)
recall_forecaster = round(
    cm_sequence_model.loc['churn', 'churn'] / 
    cm_sequence_model.loc['churn', :].sum(), 2)

cm_sequence_model
```





  <div id="df-3fd2d20a-4b29-4a63-82dd-ebf22be50283">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>retain</th>
      <th>churn</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>retain</th>
      <td>3435</td>
      <td>231</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>0</td>
      <td>462</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3fd2d20a-4b29-4a63-82dd-ebf22be50283')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3fd2d20a-4b29-4a63-82dd-ebf22be50283 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3fd2d20a-4b29-4a63-82dd-ebf22be50283');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
cm_tabular_model = pd.crosstab(
    df_lt_result.loc[df_lt_result['time_horizon']==6, 'churn'].reset_index(drop=True), 
    df_lt_result.loc[df_lt_result['time_horizon']==6,'predicted_churn'].reset_index(drop=True),
    rownames=['Actual'], 
    colnames=['Predicted'], 
    margins=False
    ).rename({0: 'retain', 1:'churn'}
    ).rename({0: 'retain', 1:'churn'}, axis=1)

precision_tabular = round(
    cm_tabular_model.loc['churn', 'churn'] / 
    cm_tabular_model.loc[:, 'churn'].sum(), 2)
recall_tabular = round(
    cm_tabular_model.loc['churn', 'churn'] / 
    cm_tabular_model.loc['churn', :].sum(), 2)

cm_tabular_model
```





  <div id="df-076f7655-0c5c-44ff-8b6c-6214d46d82b2">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>retain</th>
      <th>churn</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>retain</th>
      <td>2615</td>
      <td>214</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>99</td>
      <td>219</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-076f7655-0c5c-44ff-8b6c-6214d46d82b2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-076f7655-0c5c-44ff-8b6c-6214d46d82b2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-076f7655-0c5c-44ff-8b6c-6214d46d82b2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
print("** Comparison of two approaches\n")
print(f"Precision:\n    \
Binary Classification {precision_tabular}\
\n    Sequential {precision_forecaster}\n\n")
print(f"Recall:\n    \
Binary Classification {recall_tabular}\
\n    Sequential {recall_forecaster}")
```

    ** Comparison of two approaches
    
    Precision:
        Binary Classification 0.51
        Sequential 0.67
    
    
    Recall:
        Binary Classification 0.69
        Sequential 1.0


Comparison of two approaches show that sequential model trained with AutoML Forecasting yield even better results than model trained with AutoML Tabular. That is, sequence model shows values of 0.67 Precision and 1.00 Recall while binary classification approach shows values of 0.51 Precision and 0.69 Recall. One of the reasons could be that tabular model was trained on more imbalanced dataset because it predicts values for six months in advance, while sequence model predicts values for just 6th month in a future. Nevertheless, this exercise shows that it is recommended to try both approaches in order to evaluate which one will bring better results.

## Uplift modeling

Uplifting model is used to predict how much gain in probability we can get if we offer users some treatment in comapre to scenario without treatment. For this case, we are going to use once more time AutoMLTabular (Vertex AI) service. Like in previous cases, data is imbalanced with smaller number of users that retain after teatment is subscribed. The final goal of this section is to find treatemnts that generate highest gain in probability of retaining each individual users.


```python
COLUMN_SPEC.update({'treatment':'categorical'})
target_column = 'churn_after_treatment'
```

### Train model


```python
model_uplift_id = config['artifacts']['model_id']['uplift_model']
model_uplift_name = "uplift-churn-prediction-model"

if model_uplift_id is None:
  job_tabular = aiplatform.AutoMLTabularTrainingJob(
      display_name=f'job_train_model-{model_uplift_name}-{TIMESTAMP}',
      optimization_prediction_type='classification',
      optimization_objective='maximize-au-prc',
      column_specs=COLUMN_SPEC
  )

  model_uplift = job_tabular.run(
      dataset = DATASET_ACCOUNT,
      target_column = target_column,
      training_fraction_split = 0.7,
      validation_fraction_split = 0.15,
      test_fraction_split = 0.15,
      budget_milli_node_hours=1000,
      model_display_name=model_uplift_name,
  )
else:
  model_uplift = aiplatform.Model(model_uplift_id)
```

### Analyse model


```python
_ = model_evaluation(model_uplift, highlight_feature='treatment')
```

    Model metrics:
    
     - Area under Precision-Recall Curve: 0.97037226
     - Area under Receiver Operating Characteristic Curve: 0.968764
     - Log-loss: 0.22351304
     - Precision: 0.72
     - Recall: 0.73
    
    Confusion matrix:
    
            retain  churn
    retain   869.0   69.0
    churn     68.0  181.0



    
![png](output_86_1.png)
    


Feature importance graph shows that treatment feature are one of the most important ones. Therefore, model is relatively sensitivity towards this feature, which allows us to use it for defining uplift gains of different scenarious.

### Made predictions

In order to find best strategies, we are going to calculate gain in probability of users retaining by subtractig probability of treatment scenario with case when treatment is not offered. Hence, first we populate treatment feature with None values to calculate predictions for non treatment scenario, and after that we are going to repeat the same for each strategy but this time populating treatment feature with the name of relevant treat. 


```python
# for name in ('none', 'discount', 'free_device', 'upg_packet'):
#     df = read_csv_from_bucket(CSV_PREDICT_ACCOUNT)
#     df['treatment'] = name
#     write_csv_to_bucket(df.to_csv(index=False), f"account_predict_uplift_{name}.csv")
```


```python
model_uplift_paths_outputs = config['data']['output_data']['files_paths_uplift_model']

dfs_uplift_result_list = []

for name in model_uplift_paths_outputs.keys():

    model_uplift_path = model_uplift_paths_outputs[name]

    csv_file = os.path.join(
        "gs://", BUCKET_NAME, config['data']['input_data']['files_names_uplift_model'][name])

    DATA_UPLIFT_FOLDER = f"mn-model-uplift-{name}-output"
    gsc_output_folder = os.path.join("gs://", BUCKET_NAME, DATA_UPLIFT_FOLDER, f"predictions-uplift-{TIMESTAMP}")

    df_uplift_result = get_df_from_batch_predict(
        csv_file, 
        gsc_output_folder, 
        model_uplift_name,
        BUCKET_NAME,
        model_uplift,
        model_uplift_path)

    dfs_uplift_result_list.append(df_uplift_result)

df_uplift = pd.concat(dfs_uplift_result_list)
```


```python
# for each user extract information about probability of retain after each treatment
df_treatment = df_uplift.groupby(
    ['uuid', 'treatment'], as_index=False)['churn_after_treatment_0_scores'].first().rename(
    columns={'churn_after_treatment_0_scores': 'treatment_prob_of_retention'})

# extract no treatment cases
df_no_treatment = df_uplift.loc[df_uplift['treatment'].isna(), ['uuid',	'churn_after_treatment_0_scores']].rename(
    columns={'churn_after_treatment_0_scores': 'no_treatment_prob_of_retention'})

# compare treaetment with no treatment cases
df_compare_treatment = df_treatment.merge(df_no_treatment, on=['uuid'])
df_compare_treatment['difference'] = df_compare_treatment['treatment_prob_of_retention'] - \
    df_compare_treatment['no_treatment_prob_of_retention']

df_uplift_uid = df_uplift.loc[:, 
      ['uuid', 'treatment', 'churn_after_treatment_0_scores']
    ].fillna('None').sort_values(
      ['treatment', 'uuid']
    ).pivot_table(
      index = ['uuid'], 
      columns = 'treatment', 
      values = 'churn_after_treatment_0_scores'
    )

# compute uplift between adjacent values of treatments in a table
for col_next, col_prev in zip(df_uplift_uid.columns[:0:-1], df_uplift_uid.columns[-2::-1]): #[:-1:][::-1]):
    df_uplift_uid[col_next] = df_uplift_uid[col_next] - df_uplift_uid[col_prev]
```

Selecting best treatments for each user.
- First, filter out users that probably doesn't need any treatment, which will most certainly retain
- Second, filter out users that are lost cause, that probably could not be retain or it would be too expensive.


```python
df_highest_gain_treatment = df_compare_treatment.sort_values(
        by=['uuid', 'difference'], ascending=False
    ).loc[
        lambda df: df['difference']>=0
    ].drop_duplicates(['uuid'])

df_highest_gain_treatment.loc[
    lambda df: (df['no_treatment_prob_of_retention'] <0.95) & 
    (df['treatment_prob_of_retention']>0.2), ['uuid','treatment']].sample(10)
```





  <div id="df-e1c56521-cb00-4e95-910c-df6ad83c6dd1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uuid</th>
      <th>treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>521</th>
      <td>8b0baaba-fde5-11ec-bf37-0242ac1c0002</td>
      <td>upgrade_package</td>
    </tr>
    <tr>
      <th>732</th>
      <td>8b0f8d7e-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>8b197816-fde5-11ec-bf37-0242ac1c0002</td>
      <td>upgrade_package</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>8b1973f2-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>623</th>
      <td>8b0d82fe-fde5-11ec-bf37-0242ac1c0002</td>
      <td>upgrade_package</td>
    </tr>
    <tr>
      <th>813</th>
      <td>8b1027fc-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>1368</th>
      <td>8b176bac-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>1386</th>
      <td>8b17bff8-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>8b17322c-fde5-11ec-bf37-0242ac1c0002</td>
      <td>discount</td>
    </tr>
    <tr>
      <th>914</th>
      <td>8b11bdec-fde5-11ec-bf37-0242ac1c0002</td>
      <td>upgrade_package</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e1c56521-cb00-4e95-910c-df6ad83c6dd1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e1c56521-cb00-4e95-910c-df6ad83c6dd1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e1c56521-cb00-4e95-910c-df6ad83c6dd1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
import waterfall_chart

uuid = df_highest_gain_treatment.iloc[0, 0] 
waterfall_chart.plot(df_uplift_uid.loc[uuid].index, 
                      df_uplift_uid.loc[uuid].values)
plt.show()
```


    
![png](output_95_0.png)
    


### Budget optimization

At the end, if there is constrained budget for each of treatment, we may implement MIP (mixed integer programming) to optimise spendings regarding the expected gain from uplift outcomes. This is out of the scope of this notebook, but once we got all predictins we may consider to optimise following setup


$$
max \sum\sum X_{j,i}*P_{j,i}
$$

subject to:

> $$B_{discount} > \sum X_{discount,i}*costTreatments_{discount}$$
> $$B_{upgrade} > \sum X_{upgrade,i}*costTreatments_{upgrade}$$
> $$B_{freedevice} > \sum X_{freedevice,i}*costTreatments_{freedevice}$$
> $$X_{discount,i} + X_{free device,i} + X_{upgrade,i} + X_{none,i} = 1$$
> $$ 𝑃_{𝑗,𝑖} \in [ \, 0,1 ] \,$$
> $$X_{j,i} \in \{ 0,1 \}$$
> $$j \in \{ freedevice, upgrade, discount, none \}$$
> $$i \in [ \, 1 ... numberOfUsers ] \,$$





where


- $X_{j,i}$ - represent decision binary varaibale
- $P_{j,i}$ - probability of non churning for user i if we implement treatment j
- $costTreatments_j$ - cost of implementing treatment j
- $B_j$ - budget for treatment j

## Advance: Identify of triggers

Identification of churn triggers in a system could lead to numerous benefits, likewise: determining bottlenecks in user funnel, detecting counterintuitive parts of the product, steep learning curve, unresponsive customer support, bad quality of product etc. In fact, churn analytics could identify trigger parts of the system that need to be improved. Here we will summarize the approach of revealing users' triggers in a system through implementation of sentimental models.

### Sentiment model

Creation of an user-generated  synthetic dataset with enough data to train the model is a challenging problem itself and therefore we decided to use publicly available “Crowdflower Claritin-Twitter” dataset for this purpose. This type of approach uses similar logic as transfer learning, but in this case we are not taking weights of models but dataset of similar context.


```python
CSV_SENTIMENT_FILE =  config['data']['input_data']['file_path_sentiment_train']
SENTIMENT_MAX = 4
```


```python
model_sentiment_id = config['artifacts']['model_id']['sentiment_model']
model_sentiment_name = "sentiment-claritin-demo-model"

if model_uplift_id is None:

    sentiment_dataset = aiplatform.TextDataset.create(
        display_name="Crowdflower Claritin-Twitter" + "_" + TIMESTAMP,
        gcs_source=[CSV_SENTIMENT_FILE],
        import_schema_uri=aiplatform.schema.dataset.ioformat.text.sentiment,
    )

    job = aiplatform.AutoMLTextTrainingJob(
      display_name=model_sentiment_name+"-" + TIMESTAMP,
      prediction_type="sentiment",
      sentiment_max=SENTIMENT_MAX,
    )

    model_sentiment = job.run(
      dataset=sentiment_dataset,
      model_display_name=model_sentiment_name+"-" + TIMESTAMP,
      training_fraction_split=0.8,
      validation_fraction_split=0.1,
      test_fraction_split=0.1,
    )
else:
    model_sentiment = aiplatform.Model(model_sentiment_id)
```

Once we trained model we may use it to filter out the good from bad sentiment. For this purposes, we are going to use synthetically generated users email written customer support of imaginary telco company.


```python
input_file = os.path.join("gs://", BUCKET_NAME, config['data']['input_data']['file_name_sentiment_predict'])
output_file = config['data']['output_data']['file_path_sentiment']
```


```python
if not output_file:
    DATA_SENTIMENT_FOLDER = config['artifacts']['model_path']['sentiment_model_folder']
    gsc_output_folder = os.path.join("gs://", BUCKET_NAME, DATA_SENTIMENT_FOLDER, f"predictions-sentiment-{TIMESTAMP}")


    batch_predict_job = model_sentiment.batch_predict(
              gcs_source=input_file,
              instances_format="jsonl",
              gcs_destination_prefix=gsc_output_folder,
              predictions_format="jsonl",
              job_display_name=f"job-batch_predict-{model_sentiment_name}-{TIMESTAMP}",
              sync=True
            )
    batch_predict_job.wait()

    for row in batch_predict_job.iter_outputs():
        output_file = row.name

```


```python
json_data = bucket_to_bytes(output_file, bucket_name=BUCKET_NAME)
df_sentiment_output = pd.read_json(path_or_buf=json_data, lines=True)
df_sentiment_output['prediction'] = df_sentiment_output['prediction'].apply(lambda x: x['sentiment'])

for i in range(df_sentiment_output.shape[0]):
    txt_data = bucket_to_bytes(file_name=df_sentiment_output.iloc[i][0]['content'])
    df_sentiment_output.iloc[i, 0] = txt_data.read().decode('UTF-8')
```

Results of selected emails with bad sentiment are following:


```python
df_sentimental_select = df_sentiment_output.loc[lambda df: df['prediction']>2, ['instance']]
for i in range(df_sentimental_select.shape[0]):
    print(f"Email {i}: ", df_sentimental_select.iloc[i, 0])
```

    Email 0:  Hi, I have been a customer on your internet service for the past 12 months and I am very disappointed in the service that you are providing. I understand that there are other companies that provide internet services at a cheaper rate. I was wondering what you can do to change this? Thank you for your time,
    Email 1:  Dear service, I am writing to you because I was recently charged $400 for a phone that I purchased from you. This is significantly more than the advertised price and I'm not happy with this purchase. I would like to request that my order be canceled and my money refunded to me. Thank you,
    Email 2:  Hi service, I am writing to you because I am very unhappy with your product. I found it much too expensive and the cost is not worth what you are giving. I have found other websites that are cheaper and offer more features. I would like a refund as my experience has been very disappointing and I do not plan to purchase this again. Thank you for your time. I look forward to hearing from you soon,
    Email 3:  Dear service, I am writing to express my dissatisfaction with the pricing of your product. I found it a little expensive for my budget. The internet price seemed to be cheaper than yours. I wanted to know if you would consider lowering the price for me, or if you have any other deals for me? Thank you!


Next we are going to label our dataset of bad sentiment by comparing vector representation of words from emails with vector representation of triggers. In general, for this purposes we should use more sofisticate metrics and to filter out stop words, but just for sace of simplicity we are going to keep it simple and only count for words that have higher similarity than 0.65 to trigger words.


```python
import spacy

nlp = spacy.load('en_core_web_md')
triggers = ['price', 'quality']
trigger_tokens = nlp('price quality')
symilarity_threshold = 0.65
results = {}

for trigger in trigger_tokens:
    score = 0
    words = 0
    df_sentimental_select[trigger.text] = 0
    for i in range(df_sentimental_select.shape[0]):
        
        tokens = nlp(df_sentimental_select.iloc[i,0])

        for token in tokens:
            similarity_score = token.similarity(trigger)
            if  similarity_score > symilarity_threshold:
                score += similarity_score
                words += 1
        df_sentimental_select.loc[lambda df: df.index[i],trigger.text]  = round(score/words,2) if words>0 else 0.

df_sentimental_select[triggers] = df_sentimental_select[triggers].div(df_sentimental_select[triggers].sum(axis=1), axis=0).fillna(0).round(2)
df_sentimental_select['external'] = 1 - df_sentimental_select[triggers].sum(axis=1)
```

As result, outcome of labeling are following:


```python
df_sentimental_select
```





  <div id="df-7ad5e9ef-67f3-41a6-8cb4-1b7d1e90aa4d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instance</th>
      <th>price</th>
      <th>quality</th>
      <th>external</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hi, I have been a customer on your internet se...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dear service, I am writing to you because I wa...</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hi service, I am writing to you because I am v...</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dear service, I am writing to express my dissa...</td>
      <td>0.57</td>
      <td>0.43</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7ad5e9ef-67f3-41a6-8cb4-1b7d1e90aa4d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7ad5e9ef-67f3-41a6-8cb4-1b7d1e90aa4d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7ad5e9ef-67f3-41a6-8cb4-1b7d1e90aa4d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python

```
