from clearml import Task, Dataset

p = Dataset.get(dataset_id="ae8c12c33b324947af9ae6379d920eb8").get_local_copy()
print('Dataset загружен ', p)