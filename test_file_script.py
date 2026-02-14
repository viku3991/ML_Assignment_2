import pandas as pd

df = pd.read_csv("breast_cancer.csv")
df.drop(["id", "diagnosis"], axis=1).iloc[:40].to_csv("test_data.csv", index=False)
