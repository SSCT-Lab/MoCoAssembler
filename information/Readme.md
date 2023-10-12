## Information File Detail
> The similarity calculation results of the same frameworks and across frameworks.


### api_info
跨框架API之间的映射信息，包含`function`、`descp`、`params`，用于进行**语义一致性变异**

### framework/layer_info
每个框架的具体函数信息，包括`API`、`descp`、`parameter_list`，用于`Parameter Change`、`Boundary Checking`和`Similarity calculation`

### framework/api_similarity
每个框架中的函数相似度，用于`API replacement`

### function.csv
存储的是抽象函数与具体函数之间的映射

| id | abs_function | torch | ms | paddle | jittor  | tf |
|----|--------------|-------|----|--------|---------|----|