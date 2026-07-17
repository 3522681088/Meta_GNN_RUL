import random

class TaskSampler:
    def __init__(self, task_loaders, tasks_per_batch=None):
        self.task_loaders = task_loaders
        self.tasks_per_batch = tasks_per_batch or len(task_loaders)
    def sample(self):
        names = list(self.task_loaders)
        chosen = random.sample(names, min(self.tasks_per_batch, len(names)))
        return [(name, self.task_loaders[name]) for name in chosen]

