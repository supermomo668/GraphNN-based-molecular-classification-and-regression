# Designing selective kinase inhibitors
### PostEra

# Usage
#### via Docker
* To run with docker and create running docker container, first run the following in working directory ```postera_ml_challenge/```:
```
docker-compose build
```
or directly to running the container as well
```
docker compose up
* To use the application where the python entrypoint is already configured, you may run a basic training via the command:
```
docker run -dit --name main -it --rm postera:latest -d data/kinase_JAK_.8Train.csv train
```
* The application has the following mode of train/evaluate (single/cross) / infer, which you may make use of using 
```
docker run -dit --name main -it --rm postera:latest -d [path/to/input/data] [train/evaluate/]
```
    or override the default entry point to run within the terminal:
```
docker run -it --user matt --name postera --entrypoint /bin/bash postera:latest
```

To test the application with the container, we may run:
```
docker exec -it postera /bin/bash
```
To run application test for basic functionality test:
```
docker exec myubuntu bash -c "pytest test.py"
```
#### Application Notes
* The detail functions of the application can be viewed with:
```
python main -h
```
which will display the pipeline
```
usage: main.py [-h] -d DATA_PATH [-m MODEL] [-bs BATCH_SIZE] [-nw NUM_WORKERS]
               [--model_path MODEL_PATH]
               {train,evaluate,infer} ...

positional arguments:
  {train,evaluate,infer}
                        train or test mode

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        path to data
  -m MODEL, --model MODEL
                        model to be used
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        dataloader workers
  --model_path MODEL_PATH, -pth MODEL_PATH
                        path to model file for load/save(*.pth)
```
# Future Direction & Ideas
* Due to the degree of freedom and the complex latent requirements of the tasks, there are a large number of potential direction to further the effort: 
    * General modelling
        * Heterogeneneous graph formulation
        * Hyperparameters refinement: balance oversmoothing and representation generation
        * Graph layers crossover (GCN/GIN)
        * Attention on aggregation 
    * Data Representation
        * Physical simulation as data generator: dynamics requirement
          * RL
        * Deeper understanding of domain thus relevant data for representation
        * Functional Groups
    * Training
        * Use modularize frame (pytorch_lightning) to automate training configurations
        * Increase diversity of metrics to troubleshoot performance
        * Increase scale of training & parameter search / normalization methods
        * Transfer learning from related dataset
        * Multi-task training
            * RL & intramolecular prediction
    * Deployment
        * Leverage high-abstraction framework to automate metrics logging/monitoring/training -> focus on key design components
        * Rapidly increase versatility/configurability of the pipline & process
        * Increase hierachy for code modularization
        * DocString, Documentation
        * Docker multi-stage build