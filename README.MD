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

python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```
**FIXME**

Example:

```python

!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --patch-size 5 --image-size 150 --lr 0.0001 --epochs 200 --num-heads 12 

``` 
**FIXME**

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- ...

## III. Predict Process

local:
```bash
python predict.py
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




