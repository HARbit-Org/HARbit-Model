def serialize_to_tfrecords_fixed(X_raw, X_features, y, subjects, output_dir="tfrecords_data", 
                                samples_per_file=1000):
    """
    Serializa datos a TFRecords para evitar problemas de memoria - VERSIÓN CORREGIDA
    """
    
    print(f"🔄 Serializando {len(X_raw)} muestras a TFRecords...")
    print(f"  🔍 Verificando tipos de datos...")
    print(f"    X_raw: {X_raw.shape} {X_raw.dtype}")
    print(f"    X_features: {X_features.shape} {X_features.dtype}")
    print(f"    y: {len(y)} elementos, tipo: {type(y[0])}, ejemplo: {y[0]}")
    print(f"    subjects: {len(subjects)} elementos, tipo: {type(subjects[0])}, ejemplo: {subjects[0]}")
    
    # CORRECCIÓN 1: Validar y convertir tipos
    # Encodificar labels si son strings
    from sklearn.preprocessing import LabelEncoder
    
    if isinstance(y[0], (str, np.str_)):
        print("  🔄 Encodificando labels de string a int...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"  📊 Labels únicos: {label_encoder.classes_}")
    else:
        y_encoded = np.array(y, dtype=np.int64)
        label_encoder = None
    
    # CORRECCIÓN 2: Convertir subjects a strings
    if isinstance(subjects[0], (int, np.integer, float, np.floating)):
        print("  🔄 Convirtiendo subjects numéricos a strings...")
        subjects_str = [str(s) for s in subjects]
    else:
        subjects_str = [str(s) for s in subjects]  # Asegurar que sean strings
    
    print(f"  ✅ Datos convertidos correctamente")
    
    # Crear directorio
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Limpiar archivos anteriores
    for file in output_dir.glob("*.tfrecord"):
        file.unlink()
    
    # Función para crear ejemplo de TFRecord
    def create_tf_example(raw_data, features, label, subject):
        """Convierte una muestra a tf.train.Example"""
        
        # Convertir a bytes
        raw_bytes = tf.io.serialize_tensor(raw_data.astype(np.float32)).numpy()
        features_bytes = tf.io.serialize_tensor(features.astype(np.float32)).numpy()
        
        # CORRECCIÓN: Asegurar tipos correctos
        label_int = int(label) if not isinstance(label, int) else label
        subject_str = str(subject) if not isinstance(subject, str) else subject
        
        feature = {
            'raw_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_bytes])),
            'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features_bytes])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_int])),
            'subject': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_str.encode('utf-8')]))
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Escribir por lotes
    n_files = (len(X_raw) + samples_per_file - 1) // samples_per_file
    
    for file_idx in range(n_files):
        start_idx = file_idx * samples_per_file
        end_idx = min(start_idx + samples_per_file, len(X_raw))
        
        filename = output_dir / f"data_{file_idx:04d}.tfrecord"
        
        with tf.io.TFRecordWriter(str(filename)) as writer:
            for i in range(start_idx, end_idx):
                tf_example = create_tf_example(
                    X_raw[i], X_features[i], y_encoded[i], subjects_str[i]
                )
                writer.write(tf_example.SerializeToString())
        
        print(f"  ✅ Archivo {file_idx+1}/{n_files}: {end_idx-start_idx} muestras")
    
    # Guardar metadatos
    metadata = {
        'n_samples': len(X_raw),
        'raw_shape': X_raw.shape[1:],
        'features_shape': X_features.shape[1:],
        'n_classes': len(np.unique(y_encoded)),
        'n_files': n_files,
        'samples_per_file': samples_per_file,
        'label_encoder': label_encoder.classes_.tolist() if label_encoder else None,
        'unique_subjects': list(set(subjects_str))
    }
    
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                  for k, v in metadata.items()}, f, indent=2)
    
    # Guardar el label encoder si existe
    if label_encoder:
        import joblib
        joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
        print(f"  💾 Label encoder guardado")
    
    print(f"🎯 Serialización completa: {n_files} archivos en {output_dir}")
    return str(output_dir), metadata


def create_tfrecord_dataset(tfrecord_dir, batch_size=32, combination_mode='hybrid',
                           importance_ratio=0.25, buffer_size=1000, 
                           prefetch_size=tf.data.AUTOTUNE):
    """
    Crea dataset TensorFlow que combina raw + features on-the-fly
    """
    
    print(f"🔄 Creando dataset TensorFlow desde {tfrecord_dir}")
    
    # Cargar metadatos
    import json
    with open(Path(tfrecord_dir) / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Obtener archivos TFRecord
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord"))
    tfrecord_files = [str(f) for f in sorted(tfrecord_files)]
    
    print(f"  📁 Archivos encontrados: {len(tfrecord_files)}")
    print(f"  📊 Modo de combinación: {combination_mode}")
    
    # Definir esquema de parsing
    feature_description = {
        'raw_data': tf.io.FixedLenFeature([], tf.string),
        'features': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.int64),
        'subject': tf.io.FixedLenFeature([], tf.string)
    }
    
    def parse_tfrecord(example_proto):
        """Parse un ejemplo de TFRecord"""
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Deserializar tensores
        raw_data = tf.io.parse_tensor(parsed['raw_data'], out_type=tf.float32)
        features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
        label = parsed['label']
        subject = parsed['subject']
        
        # Reshape según metadatos
        raw_shape = metadata['raw_shape']
        features_shape = metadata['features_shape']
        
        raw_data = tf.reshape(raw_data, raw_shape)
        features = tf.reshape(features, features_shape)
        
        return raw_data, features, label, subject
    
    def combine_on_the_fly(raw_data, features, label, subject):
        """Combina raw + features on-the-fly según el modo"""
        
        if combination_mode == 'hybrid':
            # MEJOR OPCIÓN: Mantener separados para modelo híbrido
            return (raw_data, features), label
            
        elif combination_mode == 'weighted_concat':
            # Normalizar features (aproximación sin StandardScaler completo)
            features_norm = tf.nn.l2_normalize(features, axis=0)
            
            # Calcular pesos por importancia
            n_raw_channels = tf.cast(tf.shape(raw_data)[-1], tf.float32)
            n_features = tf.cast(tf.shape(features)[0], tf.float32)
            
            raw_weight = (1.0 - importance_ratio) / n_raw_channels
            feature_weight = importance_ratio / n_features
            
            # Aplicar pesos
            raw_weighted = raw_data * raw_weight
            features_weighted = features_norm * feature_weight
            
            # Expandir features a dimensión temporal
            timesteps = tf.shape(raw_data)[0]
            features_expanded = tf.tile(
                tf.expand_dims(features_weighted, 0), 
                [timesteps, 1]
            )
            
            # Concatenar
            combined = tf.concat([raw_weighted, features_expanded], axis=1)
            return combined, label
            
        elif combination_mode == 'selective_concat':
            # Seleccionar top-K features por varianza (aproximada)
            k = tf.minimum(50, tf.shape(features)[0])  # Máximo 50 features
            
            # Selección simple por magnitud (proxy de importancia)
            _, top_indices = tf.nn.top_k(tf.abs(features), k=k)
            features_selected = tf.gather(features, top_indices)
            
            # Expandir y concatenar
            timesteps = tf.shape(raw_data)[0]
            features_expanded = tf.tile(
                tf.expand_dims(features_selected, 0),
                [timesteps, 1]
            )
            
            combined = tf.concat([raw_data, features_expanded], axis=1)
            return combined, label
        
        else:
            raise ValueError(f"Modo no soportado: {combination_mode}")
    
    # Crear dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="")
    
    # Pipeline de procesamiento
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(combine_on_the_fly, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimizaciones
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(prefetch_size)
    
    print(f"  ✅ Dataset creado: batch_size={batch_size}")
    
    return dataset, metadata