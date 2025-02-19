# lmcache-vllm-extended
This is the extended driver for LMCache to run in vLLM.

The repository contains basic code and templates for deploying the IK2221 course project.

## Required Repositories

To deploy your project, you need some other repositories that are already available on Github:

### 1. LMCache Engine ([Link][LMCache])

This repository contains the engine of LMCache for inferencing.
All the processing regarding the checks for KV-cache availability, and need for storing or retrieving chunks of data are happening here. 
The engine contains a (small) local storage to keep the KV-cache data on the same server.
This reduces the requirement for retrieving of data from the remote storage.

We use the ```v0.1.4-alpha``` version of LMCache in this project.

### 2. LMCache Server ([Link][LMCache-Server])

This repository contains all essential code for storing chunks of data (i.e., KVs) on a so called, remote server.
Ideally, this repository should be deployed on a separate server and allow multiple instances of LMCache engines to fetch and store data.
However, in this project we have only one LMCache instance and the server is co-located on the same server just for simplicity of deploying the system.

### 3. LMCache vLLM Extended (Current Repository)

This repository contains vLLM injection parts required by LMCache.
In this project we extend this to include new APIs to serve user requests as well as a basic frontend to visualize user prompts and responses.

Most of your code should be implemented here unless asked otherwise in the project description.

## Setup Server

You have received a server with GPU and Ubuntu installed to run the basic setup of the system including all three repositories.
To do so, you need to clone all mentioned repositories and follow LMCache documentation to run the system. 

To make the process a bit easier for you, there is a bash script file ```project-setup.sh``` in this repository that automatically fetches the required repositories, creates a proper python virtual environment and installs the required packages. 
Please make sure the provided script file is executed correctly without any error in case you are using that. 

Additionally, you can use the following documentation to manually setup your server in case there are some issues with the provided script file:



[LMCache]: https://github.com/LMCache/LMCache
[LMCache-Server]: https://github.com/LMCache/LMCache