from .models import build_model

def create(sensor_num,cfg): return build_model("transformer",sensor_num,cfg)
