from .models import CNNLSTM

def create(sensor_num,cfg): return CNNLSTM(sensor_num,cfg["hidden_dim"],cfg["dropout"])
