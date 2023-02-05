from joblib import load


from joblib import load
import os
print(os.listdir())
model = load( './model_joblib/OEX_ridge_base.joblib')