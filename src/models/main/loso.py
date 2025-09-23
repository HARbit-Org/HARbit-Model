from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path.cwd().parent))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import clone_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

from .callbacks import callbacks
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')

from utils import *

def loso_cross_validation(features_combined, model_architecture_func, model_params, 
                         group_size=8, step_size=1, epochs=50, batch_size=32, 
                         verbose=1, save_results=True, results_path="loso_results"):
    """
    Leave-One-Subject-Out Cross Validation para HAR
    
    Args:
        features_combined: DataFrame con características combinadas
        model_architecture_func: Función que crea el modelo (ej: create_cnn_lstm_model)
        model_params: Parámetros del modelo (input_shape, num_classes, etc.)
        group_size: Tamaño de secuencia para CNN-LSTM
        step_size: Paso para crear secuencias
        epochs: Épocas de entrenamiento
        batch_size: Tamaño de batch
        verbose: Nivel de verbosidad
        save_results: Si guardar resultados
        results_path: Ruta para guardar resultados
    """
    
    print("🚀 INICIANDO LOSO CROSS-VALIDATION")
    print("=" * 60)
    
    # Obtener usuarios únicos
    users = features_combined['Subject-id'].unique()
    n_users = len(users)
    
    print(f"👥 Total de usuarios: {n_users}")
    print(f"📊 Usuarios: {users}")
    
    # Almacenar resultados
    loso_results = {
        'user': [],
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'confusion_matrix': [],
        'classification_report': [],
        'train_samples': [],
        'test_samples': [],
        'y_true': [],
        'y_pred': []
    }
    
    all_y_true = []
    all_y_pred = []
    
    # Iterar sobre cada usuario (LOSO)
    for i, test_user in enumerate(users):
        print(f"\n🔄 FOLD {i+1}/{n_users}: Usuario de test = {test_user}")
        print("-" * 40)
        
        # División de datos: un usuario para test, resto para train
        train_data = features_combined[features_combined['Subject-id'] != test_user].copy()
        test_data = features_combined[features_combined['Subject-id'] == test_user].copy()
        
        print(f"📊 Train: {len(train_data)} muestras de {len(train_data['Subject-id'].unique())} usuarios")
        print(f"📊 Test: {len(test_data)} muestras del usuario {test_user}")
        
        # Verificar que haya suficientes datos
        if len(train_data) < 100 or len(test_data) < 10:
            print(f"⚠️ Datos insuficientes para usuario {test_user}, saltando...")
            continue
        
        try:
            # Preparar secuencias para entrenamiento
            X_train, y_train, _, le_train = prepare_features_for_cnn_lstm_sequences(
                train_data, group_size=group_size, step_size=step_size
            )
            
            # Preparar secuencias para test
            X_test, y_test, _, le_test = prepare_features_for_cnn_lstm_sequences(
                test_data, group_size=group_size, step_size=step_size
            )
            
            # Convertir etiquetas de test al espacio del train
            y_test_labels = le_test.inverse_transform(y_test)
            
            # Filtrar solo clases que existen en train
            train_classes = set(le_train.classes_)
            test_classes = set(y_test_labels)
            common_classes = train_classes.intersection(test_classes)
            
            if len(common_classes) < 2:
                print(f"⚠️ Muy pocas clases comunes para usuario {test_user}, saltando...")
                continue
            
            print(f"🎯 Clases en train: {sorted(train_classes)}")
            print(f"🎯 Clases en test: {sorted(test_classes)}")
            print(f"🎯 Clases comunes: {sorted(common_classes)}")
            
            # Filtrar datos de test para incluir solo clases comunes
            mask = np.isin(y_test_labels, list(common_classes))
            X_test_filtered = X_test[mask]
            y_test_filtered = y_test_labels[mask]
            
            # Transformar etiquetas de test usando el label encoder de train
            y_test_encoded = le_train.transform(y_test_filtered)
            
            print(f"📊 Test filtrado: {len(X_test_filtered)} muestras")
            
            # Crear y compilar modelo
            input_shape = (X_train.shape[1], X_train.shape[2])
            num_classes = len(le_train.classes_)

            model = model_architecture_func(input_shape=input_shape, num_classes=num_classes)
            
            # Entrenar modelo
            print(f"🏋️ Entrenando modelo...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0 if verbose == 0 else 1
            )
            
            # Evaluar modelo
            print(f"🔍 Evaluando modelo...")
            y_pred = model.predict(X_test_filtered, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test_encoded, y_pred_classes)
            
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_encoded, y_pred_classes, average='macro', zero_division=0
            )
            
            # Reporte de clasificación
            report = classification_report(
                y_test_encoded, y_pred_classes,
                target_names=[le_train.classes_[i] for i in sorted(np.unique(y_test_encoded))],
                output_dict=True,
                zero_division=0
            )
            
            # Matriz de confusión
            cm = confusion_matrix(y_test_encoded, y_pred_classes)
            
            # Guardar resultados
            loso_results['user'].append(test_user)
            loso_results['accuracy'].append(accuracy)
            loso_results['precision_macro'].append(precision)
            loso_results['recall_macro'].append(recall)
            loso_results['f1_macro'].append(f1)
            loso_results['confusion_matrix'].append(cm)
            loso_results['classification_report'].append(report)
            loso_results['train_samples'].append(len(X_train))
            loso_results['test_samples'].append(len(X_test_filtered))
            loso_results['y_true'].append(y_test_encoded)
            loso_results['y_pred'].append(y_pred_classes)
            
            # Acumular para métricas globales
            all_y_true.extend(y_test_encoded)
            all_y_pred.extend(y_pred_classes)
            
            print(f"✅ Usuario {test_user}: Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error con usuario {test_user}: {str(e)}")
            continue
    
    # Calcular métricas agregadas
    print(f"\n📊 RESULTADOS LOSO CROSS-VALIDATION")
    print("=" * 60)
    
    if len(loso_results['accuracy']) > 0:
        mean_accuracy = np.mean(loso_results['accuracy'])
        std_accuracy = np.std(loso_results['accuracy'])
        mean_precision = np.mean(loso_results['precision_macro'])
        mean_recall = np.mean(loso_results['recall_macro'])
        mean_f1 = np.mean(loso_results['f1_macro'])
        
        print(f"🎯 Accuracy promedio: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"🎯 Precision promedio: {mean_precision:.4f}")
        print(f"🎯 Recall promedio: {mean_recall:.4f}")
        print(f"🎯 F1-Score promedio: {mean_f1:.4f}")
        print(f"📊 Usuarios evaluados: {len(loso_results['accuracy'])}/{n_users}")
        
        # Métricas globales (concatenando todas las predicciones)
        if len(all_y_true) > 0:
            global_accuracy = accuracy_score(all_y_true, all_y_pred)
            global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(
                all_y_true, all_y_pred, average='macro', zero_division=0
            )
            
            print(f"\n🌍 MÉTRICAS GLOBALES (todos los usuarios):")
            print(f"🎯 Accuracy global: {global_accuracy:.4f}")
            print(f"🎯 Precision global: {global_precision:.4f}")
            print(f"🎯 Recall global: {global_recall:.4f}")
            print(f"🎯 F1-Score global: {global_f1:.4f}")
        
        # Guardar resultados
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # DataFrame con resultados por usuario
            results_df = pd.DataFrame({
                'user': loso_results['user'],
                'accuracy': loso_results['accuracy'],
                'precision_macro': loso_results['precision_macro'],
                'recall_macro': loso_results['recall_macro'],
                'f1_macro': loso_results['f1_macro'],
                'train_samples': loso_results['train_samples'],
                'test_samples': loso_results['test_samples']
            })
            
            results_df.to_csv(f"{results_path}_summary_{timestamp}.csv", index=False)
            
            # Guardar resultados completos
            joblib.dump(loso_results, f"{results_path}_complete_{timestamp}.joblib")
            
            print(f"💾 Resultados guardados: {results_path}_*_{timestamp}.*")
        
        return {
            'summary': results_df,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'global_accuracy': global_accuracy if 'global_accuracy' in locals() else None,
            'detailed_results': loso_results
        }
    
    else:
        print("❌ No se pudieron evaluar usuarios")
        return None

def visualize_loso_results(loso_results_dict, save_path = None):
    """
    Visualiza los resultados de LOSO cross-validation
    """
    if loso_results_dict is None:
        print("❌ No hay resultados para visualizar")
        return
    
    if save_path is None:
        save_path = r"F:\UPC\Tesis\HARbit-Model\src\figures\images\loso_results_visualization.png"

    results_df = loso_results_dict['summary']
    
    # Configurar figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy por usuario
    axes[0, 0].bar(range(len(results_df)), results_df['accuracy'], alpha=0.7)
    axes[0, 0].axhline(y=loso_results_dict['mean_accuracy'], color='red', 
                       linestyle='--', label=f"Media: {loso_results_dict['mean_accuracy']:.3f}")
    axes[0, 0].set_title('Accuracy por Usuario (LOSO)')
    axes[0, 0].set_xlabel('Usuario')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels(results_df['user'], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribución de métricas
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    means = [results_df[metric].mean() for metric in metrics]
    stds = [results_df[metric].std() for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    axes[0, 1].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
    axes[0, 1].set_title('Métricas Promedio (LOSO)')
    axes[0, 1].set_ylabel('Valor')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Tamaño de datasets
    width = 0.35
    x_pos = np.arange(len(results_df))
    axes[1, 0].bar(x_pos - width/2, results_df['train_samples'], width, 
                   label='Train', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, results_df['test_samples'], width, 
                   label='Test', alpha=0.7)
    axes[1, 0].set_title('Tamaño de Datasets por Usuario')
    axes[1, 0].set_xlabel('Usuario')
    axes[1, 0].set_ylabel('Número de Muestras')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(results_df['user'], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Boxplot de métricas
    metrics_data = [results_df[metric] for metric in metrics]
    axes[1, 1].boxplot(metrics_data, labels=['Acc', 'Prec', 'Rec', 'F1'])
    axes[1, 1].set_title('Distribución de Métricas')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Estadísticas descriptivas
    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(results_df[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].describe())