# Project Migraine: Part 2
![Example Image](figures/sentiment_distribution.png)

## 🎯Tested Environment
The project has been tested on the following configurations:
- Operating System: Ubuntu 20.04.5 LTS
- NVIDIA-SMI 525.60.13
- Driver Version: 525.60.13
- CUDA Version: 12.0

## 👩‍💻Installing Conda environment:
1. Activate Conda in bash
2. Install packages using following command:
```
$ conda env create -f installations/environment.yml
```
3. If you only want to install packages without version: ```conda create --name <env-name> --file installations/packages.txt```
4. Install any remaining packages using pip as follows:
```
$ pip install -r installations/requirements.txt
```
## 🏃Running the Classification Code
1. Clone the repositry: ```git clone https://github.com/swati-rajwal/migraine.git```
2. Get access to dataset, not publicly shared at the moment. Alternatively, you can use this pipeline to your own dataset as well. Put the dataset in ```data``` folder
3. Create dataset splits: 
```python 
python B_nfold_split.py <csv_file_path> <output_folder_path>
```
4. Run ```chmod +x C_1_run_cls_multiGPU.sh``` to ensure you have rights to run this file.
5. Run ```C_1_run_cls_multiGPU.sh``` that in turn runs the ```C_2_simpletransformers_cls.py``` file for RoBERTa based classification
6. As an example, you can run a command like ```./C_1_run_cls_multiGPU.sh &> results/roberta_run_$(date +%Y-%m-%d).log```
6. If you re-run step 4, make sure to either delete or rename the folder 'model'
7. For evaluation, run ```python D_eval_model.py task_configs/migraine.json &> results/eval_run_$(date +%Y-%m-%d).log``` command.
8. To understand the sentiment across various medication groups, run ```python E_sentiment_analysis.py```