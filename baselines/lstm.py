from .models import EncoderRegressor
from models.lstm_encoder import LSTMEncoder

def create(sensor_num,cfg): return EncoderRegressor(LSTMEncoder(sensor_num,cfg["hidden_dim"],cfg["embedding_dim"],cfg["dropout"]),cfg["embedding_dim"])
