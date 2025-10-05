from marshmallow import Schema, fields

class SensorDataResponseSchema(Schema):
    timestamp = fields.Float()
    x = fields.Float()
    y = fields.Float()
    z = fields.Float()

class MetadataSchema(Schema):
    gyro_samples = fields.Int()
    accel_samples = fields.Int()
    total_samples = fields.Int()

class ProcessedDataSchema(Schema):
    gyro = fields.List(fields.Nested(SensorDataResponseSchema))
    accel = fields.List(fields.Nested(SensorDataResponseSchema))
    metadata = fields.Nested(MetadataSchema)

class PredictionResultSchema(Schema):
    window_start = fields.Str(required=True)
    window_end = fields.Str(required=True)
    predicted_activity = fields.Str(required=True)
    model_version = fields.Str(required=True)
    created_at = fields.Str(required=True)

class DataResponseSchema(Schema):
    data = fields.List(fields.Nested(PredictionResultSchema))