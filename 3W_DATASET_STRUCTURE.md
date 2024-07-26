The 3W Dataset consists of multiple Parquet files saved in the [dataset](dataset) directory and structured as follows. 

There are two types of subdirectory: 

* The [folds](dataset/folds) subdirectory holds all 3W Dataset configuration files. For each specific project released in the 3W Project there will be a file that will specify how and which data must be loaded for training and testing in multiple folds of experimentation. This scheme allows implementation of cross validation and hyperparameter optimization by the 3W Toolkit users. In addition, this scheme allows the user to choose some specific characteristics to the desired experiment. For example: whether or not simulated and/or hand-drawn intances should be considered in the training set. It is important to clarify that specifying which instances make up which folds will always be random but fixed in each configuration file. This is considered necessary so that results obtained for the same problem with different approaches can be compared;
* The other subdirectories holds all 3W Dataset data files. The subdirectory names are the instances' labels. Each file represents one instance. The filename reveals its source. All files are standardized as follows:
    * All Parquet files are created and read with pandas functions, `pyarrow` engine and `brotli` compression;
    * For each instance, timestamps corresponding to observations are stored in Parquet file as its index and loaded into pandas DataFrame as its index;
    * Each observation is stored in a line of a Parquet file and loaded as a line of a pandas DataFrame; 
    * All variables are stored as float in columns of Parquet files and loaded as float in columns of pandas DataFrame;
    * All labels are stored as `Int64` (not `int64`) in columns of Parquet files and loaded as `Int64` (not `int64`) in columns of pandas DataFrame.