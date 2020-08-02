import pandas as pd 
from utils import *
from sklearn.pipeline import make_pipeline
if __name__ == '__main__':
    df = pd.DataFrame({'col1': [1, 2, 3, 4],
                  'col2': [5, 6, 7, 8],
                  'col3': ['A', 'B', "C", "D"],
                  'col4': ['E', 'F', 'G', 'H'],
                      })

    print(df)

    pipe_1 = make_pipeline(PandasSelector(['col1']), PandasStandardScalar())
    pipe_2 = make_pipeline(PandasSelector(['col3']), PandasOneHotEncoder())

    pre_pipe = PandasFeatureUnion([
                                    ('pipe1', pipe_1), ('pipe2', pipe_2)
                                  ])

    df_processed = pre_pipe.fit_transform(df)
    print(df_processed)