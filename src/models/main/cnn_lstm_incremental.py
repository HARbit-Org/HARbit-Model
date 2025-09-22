def adaptive_transfer_learning_existing_activities(base_model_path, target_X, target_y, 
                                                  source_le_path, target_activity_names,
                                                  validation_split=0.2, progressive_training=True):
    """
    Transfer learning para AGREGAR datos de actividades YA EXISTENTES
    
    Args:
        base_model_path: Ruta del modelo preentrenado
        target_X: Nuevos datos de entrada
        target_y: Etiquetas como strings (ej: ['Walk', 'Walk', 'Sit'])
        source_le_path: Ruta del label encoder original
        target_activity_names: Lista de actividades en los nuevos datos
        validation_split: Porcentaje para validación
        progressive_training: Si usar entrenamiento progresivo
    """
    print("🔄 TRANSFER LEARNING PARA ACTIVIDADES EXISTENTES")
    print("=" * 60)
    
    # 1. Cargar label encoder original
    import joblib
    source_le = joblib.load(source_le_path)
    original_classes = source_le.classes_
    
    print(f"📚 Clases originales del modelo: {original_classes}")
    print(f"🎯 Nuevas actividades: {target_activity_names}")
    
    # 2. Verificar compatibilidad
    unsupported_activities = [act for act in target_activity_names if act not in original_classes]
    if unsupported_activities:
        print(f"❌ Actividades no soportadas: {unsupported_activities}")
        print(f"💡 El modelo original solo soporta: {original_classes}")
        return None
    
    supported_activities = [act for act in target_activity_names if act in original_classes]
    print(f"✅ Actividades compatibles: {supported_activities}")
    
    # 3. Transformar etiquetas usando el label encoder original
    target_y_encoded = source_le.transform(target_y)
    num_original_classes = len(original_classes)
    
    print(f"📊 Dataset target: {len(target_X)} muestras")
    print(f"🎯 Manteniendo {num_original_classes} clases originales")
    
    # 4. Cargar modelo base SIN modificar la capa de salida
    base_model = load_model(base_model_path)
    print(f"✅ Modelo base cargado: {base_model.input_shape} -> {base_model.output_shape}")
    
    # 5. Decidir estrategia (más conservadora para datos existentes)
    if len(target_X) < 500:
        strategy = "feature_extraction_light"  # Solo últimas capas densas
    elif len(target_X) < 2000:
        strategy = "fine_tuning_light"  # LSTM + Dense con LR muy bajo
    else:
        strategy = "fine_tuning_full"  # Todo con LR bajo
    
    print(f"🎯 Estrategia seleccionada: {strategy}")
    
    # 6. Configurar congelamiento según estrategia
    if strategy == "feature_extraction_light":
        # Solo entrenar las últimas 2 capas densas
        for i, layer in enumerate(base_model.layers[:-2]):
            layer.trainable = False
        learning_rate = 0.0005
        
    elif strategy == "fine_tuning_light":
        # Entrenar LSTM y capas densas
        for i, layer in enumerate(base_model.layers):
            layer_type = type(layer).__name__
            if 'Conv1D' in layer_type or 'MaxPooling1D' in layer_type:
                layer.trainable = False
            else:
                layer.trainable = True
        learning_rate = 0.0001
        
    else:  # fine_tuning_full
        # Entrenar todo pero con LR muy bajo
        for layer in base_model.layers:
            layer.trainable = True
        learning_rate = 0.00005
    
    # 7. Recompilar modelo
    base_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    frozen_layers = sum(1 for layer in base_model.layers if not layer.trainable)
    trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
    
    print(f"🔒 Capas congeladas: {frozen_layers}")
    print(f"🔓 Capas entrenables: {trainable_layers}")
    print(f"📈 Learning rate: {learning_rate}")
    
    # 8. División train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        target_X, target_y_encoded,
        test_size=validation_split,
        stratify=target_y_encoded,
        random_state=42
    )
    
    # 9. Convertir a categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_original_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_original_classes)
    
    print(f"📊 División de datos:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    
    # 10. Entrenamiento conservador
    print("📚 Entrenamiento conservador para actividades existentes")
    
    # Callbacks más estrictos para evitar overfitting
    callbacks = [
        EarlyStopping(
            patience=10, 
            restore_best_weights=True,
            monitor='val_accuracy',
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            patience=5, 
            factor=0.5, 
            min_lr=1e-8,
            monitor='val_accuracy'
        )
    ]
    
    # Menos épocas para evitar overfitting
    epochs = 30 if len(target_X) < 1000 else 50
    
    history = base_model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=min(32, len(X_train) // 4),  # Batch size adaptativo
        callbacks=callbacks,
        verbose=1
    )
    
    # 11. Evaluación final
    val_loss, val_acc = base_model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\n✅ TRANSFER LEARNING COMPLETADO")
    print(f"📊 Validation Accuracy: {val_acc:.4f}")
    
    return {
        'model': base_model,
        'history': history,
        'strategy_used': strategy,
        'validation_accuracy': val_acc,
        'original_label_encoder': source_le,
        'supported_activities': supported_activities
    }

def incremental_learning_pipeline(base_model_path, source_le_path, 
                                 new_data_files, activity_mapping=None):
    """
    Pipeline para aprendizaje incremental con múltiples archivos de datos
    
    Args:
        base_model_path: Ruta del modelo base
        source_le_path: Ruta del label encoder original
        new_data_files: Lista de archivos con nuevos datos
        activity_mapping: Mapeo de nombres de actividades si es necesario
    """
    print("🔄 PIPELINE DE APRENDIZAJE INCREMENTAL")
    print("=" * 60)
    
    all_X = []
    all_y = []
    all_activities = set()
    
    # Procesar todos los archivos de nuevos datos
    for i, data_file in enumerate(new_data_files):
        print(f"\n📂 Procesando archivo {i+1}/{len(new_data_files)}: {data_file}")
        
        # Aquí cargarías y procesarías cada archivo
        # X_file, y_file = process_data_file(data_file)
        
        # all_X.append(X_file)
        # all_y.extend(y_file)
        # all_activities.update(np.unique(y_file))
    
    # Combinar todos los datos
    # X_combined = np.concatenate(all_X, axis=0)
    # y_combined = np.array(all_y)
    
    print(f"📊 Total de nuevos datos: {len(all_X)} muestras")
    print(f"🎯 Actividades encontradas: {sorted(all_activities)}")
    
    # Aplicar transfer learning
    # results = adaptive_transfer_learning_existing_activities(
    #     base_model_path=base_model_path,
    #     target_X=X_combined,
    #     target_y=y_combined,
    #     source_le_path=source_le_path,
    #     target_activity_names=list(all_activities)
    # )
    
    # return results

def compare_before_after_performance(original_model_path, updated_model_path,
                                   test_X, test_y, label_encoder):
    """
    Compara el rendimiento antes y después del transfer learning
    """
    print("📊 COMPARACIÓN DE RENDIMIENTO")
    print("=" * 40)
    
    # Cargar modelos
    original_model = load_model(original_model_path)
    updated_model = load_model(updated_model_path)
    
    # Predicciones
    y_pred_original = original_model.predict(test_X)
    y_pred_updated = updated_model.predict(test_X)
    
    y_pred_classes_original = np.argmax(y_pred_original, axis=1)
    y_pred_classes_updated = np.argmax(y_pred_updated, axis=1)
    
    # Métricas
    from sklearn.metrics import accuracy_score, classification_report
    
    acc_original = accuracy_score(test_y, y_pred_classes_original)
    acc_updated = accuracy_score(test_y, y_pred_classes_updated)
    
    print(f"🔵 Modelo original: {acc_original:.4f}")
    print(f"🟢 Modelo actualizado: {acc_updated:.4f}")
    print(f"📈 Mejora: {acc_updated - acc_original:.4f}")
    
    # Análisis por actividad
    print(f"\n📋 Reporte detallado:")
    print("Modelo actualizado:")
    print(classification_report(
        test_y, y_pred_classes_updated,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    return {
        'original_accuracy': acc_original,
        'updated_accuracy': acc_updated,
        'improvement': acc_updated - acc_original
    }