import ray

# Press the green button in the gutter to run the script.
from ray.ml import ScalingConfig
from ray.ml.preprocessors import StandardScaler
from ray.ml.train.integrations.xgboost import XGBoostTrainer
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':
    # Load data.
    data = load_breast_cancer(as_frame=True)

    dataset = ray.data.from_pandas(data['data'])

    # Split data into train and validation.
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

    # Create a test dataset by dropping the target column.
    test_dataset = valid_dataset.map_batches(
        lambda df: df.drop("target", axis=1), batch_format="pandas"
    )
    # Create a preprocessor to scale some columns.
    preprocessor = StandardScaler(columns=["mean radius", "mean texture"])

    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(
            # Number of workers to use for data parallelism.
            num_workers=2,
            # Whether to use GPU acceleration.
            use_gpu=False,
        ),
        label_column="target",
        num_boost_round=20,
        params={
            # XGBoost specific params
            "objective": "binary:logistic",
            # "tree_method": "gpu_hist",  # uncomment this to use GPUs.
            "eval_metric": ["logloss", "error"],
        },
        datasets={"train": train_dataset, "valid": valid_dataset},
        preprocessor=preprocessor,
    )
    result = trainer.fit()
    print(result.metrics)
