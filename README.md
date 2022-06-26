# ParaTra: A Parallel Transformer Inference Framework for GPUs in Edge Computing

Edge computing has been widely used for deploying and accelerating deep learning applications. Equipped with GPUs, edge nodes can process concurrent incoming inference requests of the deep learning model. However, the existing parallel processing methods for the inference tasks cannot efficiently use the GPU resources. This paper investigates the popular Transformer deep learning model and develops ParaTra, a parallel inference framework for Transformers that deployed with GPUs in edge computing. In the framework, the Transformer model is partitioned and deployed in users’ devices and the edge node to efficiently utilize their processing power. The concurrent inference tasks with different sizes are dynamically packaged in a scheduling queue and sent in batch to an encoder-decoder pipeline for processing. ParaTra can significantly reduce the overheads of parallel processing and the usage of GPU memory. Experiment results show that ParaTra can save up to 37.1% of GPU memory usage and improve 8.4 times of processing speed.

In this paper, we conducted a total of six groups of experiments to test the model partition, Online-HRRN-Packaging and pipeline, and finally compared them with multiprocess and multitask schemes at the overall level of ParaTra.

Before we start the experiment, we need to complete the deployment of the ParaTra framework by following the steps below.

Step 1. Get the source code from the following link

https://anonymous.4open.science/r/ParaTra

Step 2. Model and test data

(1) Model

Due to the limitation of github, we cannot upload the trained model, so you can refer to the following way to train and split the model.

Training Model: NMT.md

After the model is trained, the model is split using the following command.

python model_split.py

(2) Test data

Test data directory in the report directory, configured with a variety of test files of the number of entries, if necessary, can be adjusted according to the situation itself.


Step 3. Environment configuration

Installation of dependency packages


pip install -r requirements.txt

Configure GPU in config. py :

gpu_id = "0"


The code for the test is located in the ~/test folder.You can refer to the following to test it.

1. Impact of Model Partition and Deployment (Section IV.B. Table. II)

server:

python ECTrans_task.py --cmd=server --workers=1

Partitioned Model Deployment Client:

python ECTrans_task_test.py --cmd=single --embedding=0

Centralized Model Deployment Client:

python ECTrans_task_test.py --cmd=single --embedding=1


2. Impact of Scheduling Queue (Section IV.C. Table. III. Figure. 4)

(1) FCFS:

server:

python ECTrans_task.py --cmd=server --workers=1

client:

python ECTrans_task_test.py --cmd=package

(2) Online-HRRN-Packing:

server:

python python ECTrans_task_hrrn.py --cmd=server --workers=1 --workers=1

client:

python ECTrans_task_hrrn_test.py

3. Impact of Pipeline (Section IV.D. Figure. 5,6)

(1) scheduling-multitask:

python run_predict.py --num=2 --batch_size=64

--datafile=report/data_1k.txt

python run_predict.py --num=4 --batch_size=64

--datafile=report/data_1k.txt

python run_predict.py --num=6 --batch_size=64

--datafile=report/data_1k.txt

python run_predict.py --num=8 --batch_size=64

--datafile=report/data_1k.txt

python run_predict.py --num=10 --batch_size=64

--datafile=report/data_1k.txt


(2) scheduling-multiprocess:

python predict_process_n.py --instances=2

--datafile=report/data_1k.txt --batch_size=64 

python predict_process_n.py --instances=4

--datafile=report/data_1k.txt --batch_size=64 

python predict_process_n.py --instances=8

--datafile=report/data_1k.txt --batch_size=64 

python predict_process_n.py --instances=10

--datafile=report/data_1k.txt --batch_size=8 

(3) scheduling-pipeline:

python predict_pipeline_split_n.py --instances=1

--batch_size=64 --datafile=report/data_1k.txt

python predict_pipeline_split_n.py --instances=2

--batch_size=64 --datafile=report/data_1k.txt

python predict_pipeline_split_n.py --instances=4

--batch_size=64 --datafile=report/data_1k.txt

python predict_pipeline_split_n.py --instances=5

--batch_size=8 --datafile=report/data_1k.txt

4. Overall Performance of ParaTra (Section IV.E. Figure. 7)

(1) multitask test:

python run_task.py --num=16 --datafile=8 --batch_size=1

python run_task.py --num=32 --datafile=8 --batch_size=1

python run_task.py --num=64 --datafile=8 --batch_size=1

python run_task.py --num=128 --datafile=8 --batch_size=1

(2) multiprocess test:

server:

python ECTrans_task.py --cmd=server --workers=8 --batc_size=1

client:

python ECTrans_task_test --cmd=real

(3) ETCtransformer test:

server:

python ECTrans_framework.py --cmd=server 

--ip=127.0.0.1 --port=5550 --port_out=5560 --pipelines=5

client:

python ECTrans_task_test.py --cmd=package


