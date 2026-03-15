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
