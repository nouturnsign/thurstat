from thurstat import *

update_config(global_seed=2022)
X = UniformContinuousDistribution(a=0, b=1)
print(X.generate_random_values(10))
print(X.generate_random_values(15))