import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import yaml

# Cargar configuración
with open(r"F:\UPC\Tesis\HARbit-Model\src\models\config\hiperparameters.yaml", 'r') as file:
    config = yaml.safe_load(file)['config']

_cnn_lstm_config = config['cnn-lstm']

def create_transfer_cnn_lstm(base_model_path, new_num_classes, transfer_strategy="full"):
    """
    Crea modelo de transfer learning específico para tu arquitectura CNN-LSTM
    
    Args:
        base_model_path: Ruta del modelo preentrenado
        new_num_classes: Número de clases en el nuevo dataset
        transfer_strategy: 'feature_extraction', 'fine_tuning', 'full'
    """
    print(f"🔄 CREANDO TRANSFER LEARNING CNN-LSTM")
    print(f"📊 Estrategia: {transfer_strategy}")
    
    # Cargar modelo base
    base_model = load_model(base_model_path)
    print(f"✅ Modelo base cargado: {base_model.input_shape} -> {base_model.output_shape}")
    
    # Analizar arquitectura del modelo base
    print(f"🔍 Analizando arquitectura ({len(base_model.layers)} capas):")
    layer_types = []
    for i, layer in enumerate(base_model.layers):
        layer_type = type(layer).__name__
        layer_types.append(layer_type)
        print(f"  {i}: {layer_type} - {getattr(layer, 'name', 'unnamed')}")
    
    # Identificar puntos de corte para diferentes estrategias
    cnn_end_idx = None
    lstm_idx = None
    dense_start_idx = None
    
    for i, layer_type in enumerate(layer_types):
        if layer_type == 'MaxPooling1D':
            cnn_end_idx = i
        elif layer_type == 'LSTM' and lstm_idx is None:
            lstm_idx = i
        elif layer_type == 'Dense' and dense_start_idx is None:
            dense_start_idx = i
    
    print(f"📍 Puntos clave: CNN hasta={cnn_end_idx}, LSTM={lstm_idx}, Dense desde={dense_start_idx}")
    
    # Estrategias de transfer learning
    if transfer_strategy == "feature_extraction":
        # Solo entrenar capas densas finales
        feature_extractor_end = dense_start_idx - 1
        freeze_until = dense_start_idx
        
    elif transfer_strategy == "fine_tuning":
        # Congelar CNN, entrenar LSTM + Dense
        feature_extractor_end = cnn_end_idx
        freeze_until = cnn_end_idx + 1
        
    elif transfer_strategy == "full":
        # Entrenar todo, pero congelar CNN inicialmente
        feature_extractor_end = lstm_idx - 1
        freeze_until = cnn_end_idx + 1
        
    else:
        raise ValueError(f"Estrategia no válida: {transfer_strategy}")
    
    # Crear feature extractor
    feature_extractor = Model(
        inputs=base_model.input,
        outputs=base_model.layers[feature_extractor_end].output
    )
    
    # Congelar capas según estrategia
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    frozen_layers = sum(1 for layer in base_model.layers if not layer.trainable)
    trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
    
    print(f"🔒 Capas congeladas: {frozen_layers}")
    print(f"🔓 Capas entrenables: {trainable_layers}")
    
    # Crear nueva cabeza de clasificación
    x = feature_extractor.output
    
    # Si cortamos antes de las capas densas, necesitamos recrearlas
    if transfer_strategy == "feature_extraction":
        # Usar las capas LSTM y Dense existentes hasta el penúltimo Dense
        for i in range(feature_extractor_end + 1, len(base_model.layers) - 1):
            layer = base_model.layers[i]
            if hasattr(layer, 'trainable'):
                layer.trainable = True
            x = layer(x)
        
        # Solo reemplazar la capa final
        x = Dense(new_num_classes, activation='softmax', name='new_classifier')(x)
        
    else:
        # Para fine_tuning y full, agregar nuevas capas densas
        if transfer_strategy == "fine_tuning":
            # Continuar desde la salida del LSTM
            lstm_layer = base_model.layers[lstm_idx]
            x = lstm_layer(x)
        
        # Agregar capas densas personalizadas
        x = Dense(256, activation='relu', name='new_dense_256')(x)
        x = BatchNormalization(name='new_bn_256')(x)
        x = Dropout(0.3, name='new_dropout_256')(x)
        
        x = Dense(128, activation='relu', name='new_dense_128')(x)
        x = BatchNormalization(name='new_bn_128')(x)
        x = Dropout(0.3, name='new_dropout_128')(x)
        
        x = Dense(64, activation='relu', name='new_dense_64')(x)
        x = Dropout(0.3, name='new_dropout_64')(x)
        
        x = Dense(new_num_classes, activation='softmax', name='new_classifier')(x)
    
    # Crear modelo final
    transfer_model = Model(inputs=feature_extractor.input, outputs=x)
    
    # Compilar con learning rate adaptado
    lr_multipliers = {
        "feature_extraction": 0.001,
        "fine_tuning": 0.0005,
        "full": 0.0001
    }
    
    learning_rate = lr_multipliers[transfer_strategy]
    
    transfer_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Modelo de transfer learning creado")
    print(f"📊 Learning rate: {learning_rate}")
    print(f"🎯 Clases de salida: {new_num_classes}")
    
    return transfer_model, feature_extractor, {
        'strategy': transfer_strategy,
        'frozen_layers': frozen_layers,
        'trainable_layers': trainable_layers,
        'learning_rate': learning_rate
    }

def progressive_unfreezing_cnn_lstm(model, X_train, y_train, X_val, y_val, 
                                   original_architecture_info, epochs_per_stage=15):
    """
    Descongelamiento progresivo específico para arquitectura CNN-LSTM
    """
    print("🎓 DESCONGELAMIENTO PROGRESIVO CNN-LSTM")
    print("=" * 50)
    
    # Identificar capas por tipo
    layer_groups = {
        'cnn_layers': [],
        'lstm_layers': [],
        'dense_layers': []
    }
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if 'Conv1D' in layer_type or 'BatchNormalization' in layer_type and i < 10:
            layer_groups['cnn_layers'].append(i)
        elif 'LSTM' in layer_type:
            layer_groups['lstm_layers'].append(i)
        elif 'Dense' in layer_type:
            layer_groups['dense_layers'].append(i)
    
    print(f"📋 Grupos de capas identificados:")
    print(f"  CNN: {layer_groups['cnn_layers']}")
    print(f"  LSTM: {layer_groups['lstm_layers']}")
    print(f"  Dense: {layer_groups['dense_layers']}")
    
    # Etapas de descongelamiento
    stages = [
        {
            'name': 'Solo Dense',
            'unfreeze_groups': ['dense_layers'],
            'learning_rate': 0.001,
            'description': 'Entrenar solo capas densas finales'
        },
        {
            'name': 'Dense + LSTM',
            'unfreeze_groups': ['dense_layers', 'lstm_layers'],
            'learning_rate': 0.0005,
            'description': 'Añadir capas LSTM al entrenamiento'
        },
        {
            'name': 'Todo el modelo',
            'unfreeze_groups': ['dense_layers', 'lstm_layers', 'cnn_layers'],
            'learning_rate': 0.0001,
            'description': 'Fine-tuning completo del modelo'
        }
    ]
    
    histories = []
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n🔄 ETAPA {stage_idx + 1}: {stage['name']}")
        print(f"📝 {stage['description']}")
        
        # Congelar todas las capas primero
        for layer in model.layers:
            layer.trainable = False
        
        # Descongelar grupos específicos
        for group_name in stage['unfreeze_groups']:
            for layer_idx in layer_groups[group_name]:
                if layer_idx < len(model.layers):
                    model.layers[layer_idx].trainable = True
        
        # Actualizar learning rate
        model.optimizer.learning_rate = stage['learning_rate']
        
        # Recompilar modelo
        model.compile(
            optimizer=Adam(learning_rate=stage['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        trainable_count = sum(1 for layer in model.layers if layer.trainable)
        total_count = len(model.layers)
        
        print(f"🔓 Capas entrenables: {trainable_count}/{total_count}")
        print(f"📈 Learning rate: {stage['learning_rate']}")
        
        # Callbacks específicos para cada etapa
        callbacks = [
            EarlyStopping(
                patience=8 if stage_idx < 2 else 12,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(
                patience=4 if stage_idx < 2 else 6,
                factor=0.5,
                min_lr=1e-7,
                monitor='val_accuracy'
            )
        ]
        
        # Entrenar
        print(f"🏃 Entrenando por {epochs_per_stage} épocas...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_stage,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        histories.append(history)
        
        # Evaluar progreso
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"📊 Fin etapa {stage_idx + 1} - Val Accuracy: {val_acc:.4f}")
    
    return histories

def adaptive_transfer_learning_cnn_lstm(base_model_path, target_X, target_y, target_le,
                                       source_X=None, source_y=None, source_le=None,
                                       validation_split=0.2, progressive_training=True):
    """
    Pipeline completo adaptativo para transfer learning CNN-LSTM
    """
    print("🚀 TRANSFER LEARNING ADAPTATIVO CNN-LSTM")
    print("=" * 60)
    
    num_target_classes = len(target_le.classes_)
    
    # Decidir estrategia basada en el tamaño del dataset
    if len(target_X) < 1000:
        strategy = "feature_extraction"
    elif len(target_X) < 5000:
        strategy = "fine_tuning"
    else:
        strategy = "full"
    
    print(f"📊 Dataset target: {len(target_X)} muestras")
    print(f"🎯 Estrategia seleccionada: {strategy}")
    
    # Crear modelo de transfer learning
    transfer_model, feature_extractor, model_info = create_transfer_cnn_lstm(
        base_model_path, num_target_classes, strategy
    )
    
    # Preparar datos
    if source_X is not None and source_y is not None:
        print("🔄 Combinando datos source y target...")
        X_combined = np.concatenate([source_X, target_X], axis=0)
        y_combined = np.concatenate([source_y, target_y], axis=0)
    else:
        X_combined = target_X
        y_combined = target_y
    
    # División train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y_combined,
        test_size=validation_split,
        stratify=y_combined,
        random_state=42
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    # Convertir a categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_target_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_target_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_target_classes)
    
    print(f"📊 División de datos:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Validation: {X_val.shape}")
    
    # Entrenamiento
    if progressive_training and strategy in ["fine_tuning", "full"]:
        print("🎓 Entrenamiento progresivo activado")
        histories = progressive_unfreezing_cnn_lstm(
            transfer_model, X_train, y_train_cat, X_val, y_val_cat
        )
    else:
        print("📚 Entrenamiento estándar")
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-7)
        ]
        
        history = transfer_model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        histories = [history]
    
    # Evaluación final
    val_loss, val_acc = transfer_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ TRANSFER LEARNING COMPLETADO")
    print(f"📊 Validation Accuracy: {val_acc:.4f}")
    
    return {
        'model': transfer_model,
        'histories': histories,
        'model_info': model_info,
        'validation_accuracy': val_acc,
        'strategy_used': strategy
    }