import os
import logging
import joblib  # For saving/loading the model
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from river import tree, preprocessing, metrics
from quixstreams import Application
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WriteOptions

# Load environment variables from .env file
load_dotenv(".env")
n = 0

# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load configurations
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "172.16.2.117:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "event-battery")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "prediction-battery-RUL")
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "C:/Users/pattanapong/Desktop/IOT/random_forest_model.pkl")

INFLUX_URL = os.getenv("INFLUX_URL", "http://172.16.2.117:8085")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "eUq5Fj9zSW4lvJtuxymghnC_jJzS8EbNdFkNs_ZCYqsQx3yiHYIR2TxbM5SfpbqKRaxTZIHsJEZ8OMtKpoLv3A==")
INFLUX_ORG = os.getenv("INFLUX_ORG", "org-6510301011")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "bucket-6510301011")

print(f"{INFLUX_URL} {INFLUX_TOKEN} {INFLUX_ORG} {INFLUX_BUCKET} ")
exit
# Initialize InfluxDB client
try:
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    influx_writer = influx_client.write_api(write_options=WriteOptions(batch_size=1, flush_interval=1))
    logging.info("✅ InfluxDB client initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize InfluxDB client: {e}")
    exit(1)

# Load or initialize the online model
if os.path.exists(MODEL_LOCATION):
    model = joblib.load(MODEL_LOCATION)
    logging.info(f"✅ Loaded regression model from {MODEL_LOCATION}")
else:
    model = preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()
    logging.info(f"📦 Initialized new regression model")

# Metrics for evaluation
metric_rmse = metrics.RMSE()
metric_mae = metrics.MAE()
metric_r2 = metrics.R2()   # ✅ เพิ่ม metric R²

# Record counter for triggering model save
counter = 0

# Setup QuixStreams application
try:
    app = Application(
        broker_address=KAFKA_BROKER,
        loglevel="INFO",
        auto_offset_reset="earliest",
        state_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "state"),
        consumer_group="predict-from-kafka-onlineV19"
    )
    input_topic = app.topic(KAFKA_INPUT_TOPIC, value_deserializer="json")
    producer = app.get_producer()
    output_topic = app.topic(KAFKA_OUTPUT_TOPIC, value_serializer="json")
    logging.info(f"✅ Kafka topics configured: input='{KAFKA_INPUT_TOPIC}', output='{KAFKA_OUTPUT_TOPIC}'")
except Exception as e:
    logging.error(f"❌ Failed to setup QuixStreams application or topics: {e}")
    exit(1)

# Handler function to process each message from Kafka
def handle_message(data):
    global counter
    try:
        sensor_name = data.get("name", "")
        payload = data.get("payload", {})
        timestamp_d = data.get("__Timestamp__")
        # ดึง features จาก payload
        Max_Voltage_Discharg = payload.get("Max_Voltage_Discharg")
        Min_Voltage_Charg = payload.get("Min_Voltage_Charg")
        Discharge_Time = payload.get("Discharge_Time")
        Decrement = payload.get("Decrement")
        Time = payload.get("Time")
        Time_constant_current = payload.get("Time_constant_current")
        Charging_time = payload.get("Charging_time")

        # ตรวจสอบว่ามีค่า features หรือไม่
        if None in [
            Max_Voltage_Discharg,
            Min_Voltage_Charg,
            Discharge_Time,
            Decrement,
            Time,
            Time_constant_current,
            Charging_time
        ]:
            logging.error("❌ Missing one or more features in payload, skipping this message")
            return

        # Features dict สำหรับ model
        x = {
            "Discharge_Time": float(Discharge_Time),
            "Decrement": float(Decrement),
            "Max_Voltage_Discharg": float(Max_Voltage_Discharg),
            "Min_Voltage_Charg": float(Min_Voltage_Charg),
            "Time": float(Time),
            "Time_constant_current": float(Time_constant_current),
            "Charging_time": float(Charging_time)
        }

        # Predict RUL
        if hasattr(model, "predict_one"):  
            prediction = model.predict_one(x)
        elif hasattr(model, "predict"):  

            # กำหนด order ของ column ตามตอน train
            columns_order = [
                "Discharge_Time",
                "Decrement",
                "Max_Voltage_Discharg",
                "Min_Voltage_Charg",
                "Time",
                "Time_constant_current",
                "Charging_time"
            ]

            # แปลงทุกค่าเป็น float
            x_df = pd.DataFrame([{
                col: float(x[col]) for col in columns_order
            }], columns=columns_order)

            prediction = model.predict(x_df)[0]

        else:
            raise AttributeError("❌ Model does not support predict or predict_one")

        cycle_index = payload.get("Cycle_Index", 0)
        timestamp = datetime.now(timezone.utc)
        actual_rul = payload.get("RUL", None)

        if actual_rul is not None:
            actual_rul = float(actual_rul)
            # Update model
            if hasattr(model, "learn_one"):
                model.learn_one(x, actual_rul)

            # Update metrics
            metric_rmse.update(actual_rul, prediction)
            metric_mae.update(actual_rul, prediction)
            metric_r2.update(actual_rul, prediction)  # ✅ อัปเดต R²

            logging.info(
                f"🔍 Cycle={cycle_index}, Predict={prediction:.2f}, "
                f"Target={actual_rul}, RMSE={metric_rmse.get():.3f}, "
                f"MAE={metric_mae.get():.3f}, R²={metric_r2.get():.3f}"
            )

            counter += 1
            if counter % 10 == 0:
                joblib.dump(model, MODEL_LOCATION)
                logging.info(f"💾 Model saved after {counter} records to {MODEL_LOCATION}")
                counter = 0
        else:
            logging.warning("[Battery Dataset] RUL not found in payload")

        # Build result payload (เก็บฟีเจอร์ด้วย)
        result_payload = {
            "Discharge_Time": Discharge_Time,
            "Decrement": Decrement,
            "Max_Voltage_Discharg": Max_Voltage_Discharg,
            "Min_Voltage_Charg": Min_Voltage_Charg,
            "Time": Time,
            "Time_constant_current": Time_constant_current,
            "Charging_time": Charging_time,
            "RUL": int(actual_rul) if actual_rul is not None else None,
            "RUL_predicted": float(prediction),
            "r2": float(metric_r2.get()),   # ✅ เปลี่ยนจาก accuracy → r2
            "cycle_index": cycle_index,
            "timestamp": int(timestamp.timestamp() * 1000),
            "date": timestamp.isoformat()
        }
        result = {
            "id": data.get("id", ""),
            "name": sensor_name,
            "place_id": data.get("place_id", ""),
            "payload": result_payload,
            "__Timestamp__": timestamp_d
        }

        new_payload = json.dumps(result).encode("utf-8")

        # ส่งไป Kafka
        producer.produce(
            topic=output_topic.name,
            key=sensor_name,
            value=new_payload,
            timestamp=int(timestamp.timestamp() * 1000)
        )
        logging.info(f"[📤] Sent prediction to Kafka topic: {KAFKA_OUTPUT_TOPIC}")
        timestamp_ns = int(timestamp_d * 1_000_000)
        # เขียน prediction ไป InfluxDB
        point = (
            Point("battery_rul_prediction")
            .tag("sensor_id", data.get("id", "unknown"))
            .tag("place_id", data.get("place_id", "unknown"))
            .tag("name", sensor_name)
            .field("RUL_predicted", float(prediction))
            .field("r2", float(metric_r2.get()))   # ✅ เปลี่ยนจาก accuracy → r2
            .field("RUL", float(payload.get("RUL", None)))
            .time(timestamp_ns)
        )
        influx_writer.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        logging.info("[📊] Wrote prediction to InfluxDB")

    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")


# Apply the message handler to Kafka dataframe stream
sdf = app.dataframe(input_topic)
sdf = sdf.apply(handle_message)

# Start application
logging.info(f"Connecting to ...{KAFKA_BROKER}")
logging.info(f"🚀 Listening to Kafka topic: {KAFKA_INPUT_TOPIC}")
app.run()
