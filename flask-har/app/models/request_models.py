from marshmallow import Schema, fields, validate, ValidationError

class SensorDataSchema(Schema):
    timestamp = fields.Int(required=True)
    sensorType = fields.Int(required = True)
    x = fields.Float(required=True)
    y = fields.Float(required=True)
    z = fields.Float(required=True)

class InfoDataSchema(Schema):
    id = fields.Str(required=True)
    deviceId = fields.Str(required=True)
    timestamp = fields.Int(required=True)
    sampleCount = fields.Int(required = True)
    readings = fields.List(fields.Nested(SensorDataSchema), required=True)

class DataRequestSchema(Schema):
    userId = fields.Str(required=True)
    batches = fields.List(fields.Nested(InfoDataSchema), required=True)