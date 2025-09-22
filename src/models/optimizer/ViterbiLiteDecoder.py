import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns

class ViterbiLiteDecoder:
    """
    Decodificador Viterbi-lite para suavizar secuencias de actividades HAR
    """
    
    def __init__(self, classes, transition_penalty=2.0, self_bonus=1.0, 
                 min_duration=None, activity_specific_penalties=None):
        """
        Args:
            classes: Lista de nombres de clases ['Walk', 'Sit', 'Stand', ...]
            transition_penalty: Penalización base por cambiar de actividad
            self_bonus: Bonus por mantener la misma actividad
            min_duration: Dict con duración mínima por clase (en frames)
            activity_specific_penalties: Dict con penalizaciones específicas
        """
        self.classes = np.array(classes)
        self.n_classes = len(classes)
        self.transition_penalty = transition_penalty
        self.self_bonus = self_bonus
        
        # Duración mínima por clase (en frames de 5s)
        self.min_duration = min_duration or {
            'Walk': 3,      # 15 segundos mínimo
            'Sit': 4,       # 20 segundos mínimo
            'Stand': 2,     # 10 segundos mínimo
            'Type': 6,      # 30 segundos mínimo
            'Eat': 4,       # 20 segundos mínimo
            'Write': 4,     # 20 segundos mínimo
            'Workouts': 8,  # 40 segundos mínimo
            'Others': 1     # Sin restricción
        }
        
        # Penalizaciones específicas entre pares de actividades
        self.activity_penalties = activity_specific_penalties or self._default_penalties()
        
        # Crear matriz de transición
        self.transition_matrix = self._build_transition_matrix()
    
    def _default_penalties(self):
        """
        Penalizaciones por defecto basadas en lógica de actividades humanas
        """
        penalties = {
            # Transiciones muy improbables (penalización alta)
            ('Sit', 'Workouts'): 5.0,
            ('Workouts', 'Sit'): 3.0,
            ('Type', 'Workouts'): 4.0,
            ('Workouts', 'Type'): 4.0,
            ('Eat', 'Workouts'): 5.0,
            ('Workouts', 'Eat'): 3.0,
            
            # Transiciones moderadamente improbables
            ('Sit', 'Walk'): 1.5,
            ('Walk', 'Sit'): 1.0,
            ('Stand', 'Sit'): 1.0,
            ('Sit', 'Stand'): 1.0,
            
            # Transiciones naturales (penalización baja)
            ('Walk', 'Stand'): 0.5,
            ('Stand', 'Walk'): 0.5,
            ('Stand', 'Type'): 0.8,
            ('Type', 'Stand'): 0.8,
            ('Sit', 'Type'): 0.8,
            ('Type', 'Sit'): 0.8,
            ('Sit', 'Eat'): 0.5,
            ('Eat', 'Sit'): 0.5,
            ('Sit', 'Write'): 0.8,
            ('Write', 'Sit'): 0.8,
        }
        return penalties
    
    def _build_transition_matrix(self):
        """
        Construye matriz de log-probabilidades de transición
        """
        # Inicializar con penalización base
        transition_log_probs = np.full((self.n_classes, self.n_classes), 
                                     -self.transition_penalty)
        
        # Bonus para mantenerse en la misma actividad
        np.fill_diagonal(transition_log_probs, self.self_bonus)
        
        # Aplicar penalizaciones específicas
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for (from_cls, to_cls), penalty in self.activity_penalties.items():
            if from_cls in class_to_idx and to_cls in class_to_idx:
                i, j = class_to_idx[from_cls], class_to_idx[to_cls]
                transition_log_probs[i, j] = -penalty
        
        return transition_log_probs
    
    def decode_with_probabilities(self, probabilities):
        """
        Decodifica usando probabilidades del modelo (RECOMENDADO)
        
        Args:
            probabilities: Array (T, n_classes) con probabilidades por frame
        Returns:
            decoded_sequence: Array (T,) con clases decodificadas
            path_scores: Scores del camino óptimo
        """
        T, n_classes = probabilities.shape
        assert n_classes == self.n_classes
        
        # Convertir a log-probabilidades
        log_probs = np.log(probabilities + 1e-8)
        
        # Matrices de Viterbi
        viterbi_scores = np.full((T, n_classes), -np.inf)
        viterbi_path = np.zeros((T, n_classes), dtype=int)
        
        # Inicialización (t=0)
        viterbi_scores[0] = log_probs[0]
        
        # Forward pass
        for t in range(1, T):
            for j in range(n_classes):
                # Scores de venir de cualquier estado anterior
                transition_scores = (viterbi_scores[t-1] + 
                                   self.transition_matrix[:, j])
                
                # Mejor estado anterior
                best_prev = np.argmax(transition_scores)
                viterbi_scores[t, j] = (transition_scores[best_prev] + 
                                      log_probs[t, j])
                viterbi_path[t, j] = best_prev
        
        # Backward pass (recuperar camino óptimo)
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi_scores[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = viterbi_path[t+1, path[t+1]]
        
        # Scores del camino
        path_scores = np.array([viterbi_scores[t, path[t]] for t in range(T)])
        
        return path, path_scores
    
    def decode_without_probabilities(self, predictions):
        """
        Decodifica usando solo predicciones enteras (fallback)
        
        Args:
            predictions: Array (T,) con índices de clases predichas
        Returns:
            decoded_sequence: Array (T,) con secuencia suavizada
        """
        T = len(predictions)
        
        # Crear pseudo-probabilidades a partir de predicciones
        pseudo_probs = np.zeros((T, self.n_classes))
        for t, pred in enumerate(predictions):
            pseudo_probs[t, pred] = 0.8  # Alta confianza en la predicción
            # Distribuir resto entre otras clases
            mask = np.ones(self.n_classes, dtype=bool)
            mask[pred] = False
            pseudo_probs[t, mask] = 0.2 / (self.n_classes - 1)
        
        path, _ = self.decode_with_probabilities(pseudo_probs)
        return path
    
    def enforce_minimum_duration(self, sequence):
        """
        Aplica restricciones de duración mínima post-Viterbi
        """
        sequence = sequence.copy()
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        i = 0
        while i < len(sequence):
            current_class_idx = sequence[i]
            current_class = self.classes[current_class_idx]
            
            # Encontrar el final del segmento actual
            j = i
            while j < len(sequence) and sequence[j] == current_class_idx:
                j += 1
            
            segment_length = j - i
            min_length = self.min_duration.get(current_class, 1)
            
            # Si el segmento es demasiado corto, fusionarlo con vecinos
            if segment_length < min_length:
                # Determinar con qué vecino fusionar
                left_class = sequence[i-1] if i > 0 else None
                right_class = sequence[j] if j < len(sequence) else None
                
                # Preferir fusionar con el vecino más largo
                if left_class is not None and right_class is not None:
                    # Contar longitudes de segmentos vecinos
                    left_len = self._count_segment_length(sequence, i-1, backward=True)
                    right_len = self._count_segment_length(sequence, j, backward=False)
                    
                    replacement = left_class if left_len >= right_len else right_class
                elif left_class is not None:
                    replacement = left_class
                elif right_class is not None:
                    replacement = right_class
                else:
                    replacement = current_class_idx  # No cambiar si no hay vecinos
                
                sequence[i:j] = replacement
            
            i = j
        
        return sequence
    
    def _count_segment_length(self, sequence, start_idx, backward=False):
        """Cuenta la longitud del segmento que contiene start_idx"""
        if start_idx < 0 or start_idx >= len(sequence):
            return 0
        
        target_class = sequence[start_idx]
        length = 1
        
        if backward:
            # Contar hacia atrás
            i = start_idx - 1
            while i >= 0 and sequence[i] == target_class:
                length += 1
                i -= 1
        else:
            # Contar hacia adelante
            i = start_idx + 1
            while i < len(sequence) and sequence[i] == target_class:
                length += 1
                i += 1
        
        return length
    
    def decode_complete_pipeline(self, probabilities=None, predictions=None, 
                               apply_duration_constraints=True):
        """
        Pipeline completo de decodificación
        
        Args:
            probabilities: Array (T, n_classes) con probabilidades (preferido)
            predictions: Array (T,) con predicciones enteras (fallback)
            apply_duration_constraints: Si aplicar restricciones de duración
        """
        if probabilities is not None:
            print("🧠 Decodificando con probabilidades (Viterbi completo)")
            path, scores = self.decode_with_probabilities(probabilities)
        elif predictions is not None:
            print("🔢 Decodificando con predicciones enteras (Viterbi-lite)")
            path = self.decode_without_probabilities(predictions)
            scores = None
        else:
            raise ValueError("Debe proporcionar probabilities o predictions")
        
        if apply_duration_constraints:
            print("⏱️ Aplicando restricciones de duración mínima")
            path = self.enforce_minimum_duration(path)
        
        return path, scores
    
    def visualize_decoding(self, original_sequence, decoded_sequence, 
                          probabilities=None, save_path=None):
        """
        Visualiza el proceso de decodificación
        """
        fig, axes = plt.subplots(3 if probabilities is not None else 2, 1, 
                               figsize=(15, 10))
        
        T = len(original_sequence)
        time_axis = np.arange(T) * 5  # Assuming 5s frames
        
        # Plot 1: Secuencias original vs decodificada
        ax1 = axes[0]
        ax1.plot(time_axis, original_sequence, 'r-', label='Original', alpha=0.7, linewidth=2)
        ax1.plot(time_axis, decoded_sequence, 'b-', label='Decodificada', alpha=0.9, linewidth=2)
        ax1.set_ylabel('Clase')
        ax1.set_title('Comparación: Original vs Decodificada')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Añadir nombres de clases en y-axis
        ax1.set_yticks(range(self.n_classes))
        ax1.set_yticklabels(self.classes)
        
        # Plot 2: Matriz de transición
        ax2 = axes[1]
        sns.heatmap(self.transition_matrix, 
                   xticklabels=self.classes, 
                   yticklabels=self.classes,
                   annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Matriz de Transición (Log-Probs)')
        
        # Plot 3: Probabilidades (si están disponibles)
        if probabilities is not None:
            ax3 = axes[2]
            im = ax3.imshow(probabilities.T, aspect='auto', cmap='Blues', origin='lower')
            ax3.set_ylabel('Clases')
            ax3.set_xlabel('Tiempo (frames)')
            ax3.set_title('Probabilidades por Frame')
            ax3.set_yticks(range(self.n_classes))
            ax3.set_yticklabels(self.classes)
            
            # Colorbar
            plt.colorbar(im, ax=ax3, label='Probabilidad')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def get_model_probabilities(model, X_test):
    """
    Extrae probabilidades del modelo en lugar de solo predicciones
    """
    print("🔍 Extrayendo probabilidades del modelo...")
    
    # Obtener probabilidades softmax
    probabilities = model.predict(X_test)
    
    print(f"✅ Probabilidades extraídas: {probabilities.shape}")
    print(f"   Rango de probabilidades: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    return probabilities