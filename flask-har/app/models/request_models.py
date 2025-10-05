from marshmallow import Schema, fields, validate, ValidationError

class SensorDataSchema(Schema):
    timestamp = fields.Float(required=True)
    x = fields.Float(required=True)
    y = fields.Float(required=True)
    z = fields.Float(required=True)

class DataRequestSchema(Schema):
    gyro = fields.List(fields.Nested(SensorDataSchema), required=True)
    accel = fields.List(fields.Nested(SensorDataSchema), required=True)