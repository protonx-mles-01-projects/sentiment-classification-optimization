# Sentiment Classification Optimization

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

[Slide about project](https://docs.google.com/presentation/d/1YldvqbBbMXc0gdUIo71sTJtB87oGZTGr5eUhd1dvY1E/edit)


Authors:
- Github: @NamTran072, @maitrang-ng, @mp1704, @quangle

Advisors:
- Github: bangoc123

## I.  Set up environment
- Step 1:

```
conda env create -f environment.yml
```

- Step 2:
```
pip install -r requirements.txt
```

## II. Training Process

Training script:


```python

python train.py --epochs ${epochs} --batch-size ${batch_size} --test-size ${test_size} --learning-rate ${learning_rate} --output-dir ${output_dir} --save-model-dir ${save_model_dir}

```

Example:

```python

python train.py --epochs 2 --batch-size 16 --test-size 0.2 --learning-rate 5e-5 --output-dir ${output_dir} --save-model-dir ${save_model_dir}

``` 

There are some important arguments for the script you should consider when running it:

- `output-dir` (str): The output directory where the model predictions and checkpoints will be written.
- `save-model-dir` (str): The directory where the model is saved
- `epochs` (int, , optional, defaults to 2): Total number of training epochs to perform
- `learning-rate` (float, optional, defaults to 5e-5): The initial learning rate for AdamW optimizer.
- `batch-size` (int, optional, defaults to 16): The batch size per GPU/TPU core/CPU for training. 
- `test-size` (float, optinal, defaults to 0.2): Size of the test split represent the proportion of the dataset to include in the test split.

## III. Predict Process

local:
```bash
python predict.py --model-path ${model_path}
```

remote:

* Distilbert model:
POST http://35.188.66.210/api/v1/predict
request body:
```
{
    "review": "xxx"
}
```

response body:
```
{
    "score": 99.98,
    "sentiment": "Negative"
}
```

* ONNX model:
POST http://35.188.66.210/api/v2/predict
request body:
```
{
    "review": "xxx"
}
```

response body:
```
{
    "score": 99.98,
    "sentiment": "Negative"
}
```

## IV. Container and deployment
(Optional) download prebuilt models from: https://drive.google.com/drive/u/1/folders/1wvyNMuxkCDzJOgFw5WaBZfslzobjTimL to under `model` folder to skip training step

Build and push Docker image to Google Cloud Registry

Install Google Cloud CLI:
https://cloud.google.com/sdk/docs/install-sdk

Google Cloud Login:
```
gcloud auth login
```

Configure Docker permission first time:
```
gcloud auth configure-docker
```

Build docker image to `gcr.io` region, under `project_id`, with `folder_path` for Google Cloud Container Registry:
```
docker build ./ -f .Dockerfile -t gcr.io/project_id/folder_path:sentiment-classification-v1.0
```

Push docker image to Google Cloud Container Registry:
```
docker build ./ -f .Dockerfile -t gcr.io/project_id/folder_path:sentiment-classification-v1.0
```

In Google Cloud, go to Compute Engine, create Instance Template with Container configured using Container Registry path above: `gcr.io/project_id/folder_path:sentiment-classification-v1.0`

Create new VM Instances or VM Instance Group from above Instance Template, allow both HTTP and HTTPS traffic

## V. Result and Comparision




