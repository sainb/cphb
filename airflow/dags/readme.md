I am building recommender system for ecommerce website. I have a table with top 1 mln queries stored in hive table. Make two airflow pipelines: one for feature generation and another for CTR model inference for production setting.  The goal is to recommend top 10,000 recommended products for each query. The possible set of all products is 10 mln. The goal is to choose top 10,000 products from 10 mln for each query.

I will use AWS EMR cluster HiveOperator, SparkSQLOperator for feature engineering, and SparkOperator for making inference with xgboost model. 

I will use different embeddings to generate top K candidates for each query. In my experiment I may get 100,000 for a given query and use CTR model to further reduce it to 10,000. 

I will run 1 production version, 1 holdout version, and 5 different experiments based of this pipeline. Make it customizable to allow for all these variants generated within a single pipeline framework. I want to avoid redundant calculations of features for query, product, and query-product pair in my experiments and production pipeline.
I store predictions in hive table.