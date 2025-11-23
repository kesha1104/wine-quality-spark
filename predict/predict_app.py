import argparse
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel


def get_spark(app_name: str = "WineQualityPrediction"):
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_dataset(spark, path: str):
    return (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .csv(path)
    )


def main(args):
    spark = get_spark()

    print(f"Loading test data from: {args.test_path}")
    test_df = load_dataset(spark, args.test_path)

    print(f"Loading model from: {args.model_path}")
    model = PipelineModel.load(args.model_path)

    print("Generating predictions...")
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",          # label column created by the pipeline
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator.evaluate(predictions)
    print(f"F1 score on test dataset: {f1}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using trained wine-quality model.")
    parser.add_argument("test_path", help="Path to test CSV file.")
    parser.add_argument(
        "--model_path",
        default="./model",
        help="Path where the trained PipelineModel is saved (default: ./model)."
    )

    args = parser.parse_args()
    main(args)

