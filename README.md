Filesystem performance is highly dependent on multiple interacting configuration parameters such 
as disk block size, read-ahead settings, and I/O scheduler policies. Identifying optimal parameter 
combinations manually is challenging due to the large configuration space and complex workload 
interactions. This project explores the use of machine learning techniques to model filesystem 
performance and predict the optimal set of configuration of these tunable parameters, based on 
various performance metrics like bandwidth, latency and IOPS. A dataset of approximately 10,000 
filesystem experiments was collected and used to train multiple ensemble-based regression 
models. Several tree-based machine learning algorithms were evaluated and compared using 
standard performance metrics. Experimental results demonstrate that gradient boosting based 
models provide superior predictive accuracy for this problem. The project establishes a foundation 
for building an automated filesystem configuration optimization system.

Modern file systems expose a wide range of tunable parameters that significantly affect filesystem 
performance. Parameters such as block size, read-ahead size, commit interval, journaling mode, 
and I/O scheduling policies determine how efficiently storage devices handle different workloads. 
In real-world deployments, system administrators often rely on default settings or manual 
experimentation to tune these parameters. However, as the number of configurable parameters 
increases, identifying optimal combinations becomes increasingly difficult. This requires very 
skilled personnel and continuous monitoring as the IO workload is very heterogenous in nature. 
Machine learning provides a data-driven approach to model the relationship between filesystem 
configurations and performance outcomes. By training predictive models on experimental data, it 
becomes possible to estimate the best set of these parameters for optimum performance. 
In addition to performance optimization, practical filesystem deployment must also consider 
reliability and security constraints. Certain configuration parameters, particularly journaling 
modes, directly influence data consistency guarantees and crash recovery behavior. Therefore, this 
project introduces a security-aware optimization layer in which a user-specified security level 
constrains the configuration space explored by the optimizer. This ensures that recommended 
configurations not only maximize performance metrics but also comply with required security and 
reliability policies. 
The goal of this project is to develop a machine learning based framework capable of predicting 
filesystem performance metrics and assisting in adaptive configuration optimization for changing 
IO workloads. 
The primary objectives of this project are: 
• Collect and prepare a dataset of filesystem performance experiments. 
• Apply machine learning techniques to predict performance metrics. 
• Compare multiple regression models for prediction accuracy. 
• Identify the most effective model for filesystem performance prediction. 
