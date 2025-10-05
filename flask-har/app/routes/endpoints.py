from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from loguru import logger
from app.models.request_models import DataRequestSchema
from app.models.response_models import DataResponseSchema
from app.services.data_processor import process_data

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/predict', methods=['POST'])
def process_data_endpoint():
    try:
        # Validar datos de entrada
        schema = DataRequestSchema()
        try:
            data_request = schema.load(request.json)
        except ValidationError as err:
            logger.error(f"Error de validación: {err.messages}")
            return jsonify({'error': 'Datos inválidos', 'details': err.messages}), 422
        
        logger.info(f"Datos recibidos: {data_request}")
        
        # Procesar datos
        processed_data = process_data(data_request)
        
        # Preparar respuesta
        response_schema = DataResponseSchema()
        response_data = response_schema.dump({'data': processed_data})
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor', 'details': str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'flask-har-processor'}), 200