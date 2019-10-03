"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 24 - Marzo - 2019
Description: Validación cruzada del clasficador encargado de detectar las Glándulas.
#########################################################################################################
"""


# Utilidades:
import os
from operator import itemgetter
import numpy as np
import math
from scipy import interp
import matplotlib.pyplot as plt
from tensorflow import set_random_seed

# Construcción de los datos:
import matplotlib.image as mpimg
from libs.DataUtils import build_from_QuPath, add_scale, data_augmentation
from libs.DataUtils import zipImagesFiles, draw_patches_of_class

# Preprocesado:
from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight
from keras.utils import to_categorical

# Módulos para la red:
import keras
from models.cnn import GlandsDetector
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Métricas:
import keras_metrics as km
from sklearn.metrics import roc_curve, auc, confusion_matrix

#########################################################################################################

# Establecer la semilla aleatoria:
SEED = 12
np.random.seed(SEED)
set_random_seed(SEED)

# Determina si dibujar o no los FP y FN
define_compute_FPFN = False

#########################################################################################################

# Definir la función que aplica el learning rate decay:
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 25.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

	return lrate

#########################################################################################################

def find_optimal_cuttoff(fpr, tpr, thresholds):
    """ Devuelve el umbral de corte óptimo para la probabilidad
    a patir de la curva ROC.
    """
    optimal_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
    return thresholds[optimal_idx]


#########################################################################################################

def error_summary(y_orig, y_bin, y_pred):
    """ Muestra el porcentaje de error que comete el clasificador en
    cada clase.
    y_orig -- Las etiquetas originales.
    y_bin -- El etiquetado binario.
    y_pred -- Lo predicho por el clasificador.
    """

    print("")

    # Concatenamos los tres vectores en uno:
    data = np.c_[y_orig, y_bin, y_pred]

    # Almacenamos el número total de errores:
    N = len(list(filter(lambda a: a[0] != a[1], data[:,1:])))

    # El número total de cada clase:
    N_gland = len(list(filter(lambda a: a[0]==1 , data)))
    N_stroma = len(list(filter(lambda a: a[0]==0 , data)))
    N_contaminated = len(list(filter(lambda a: a[0]==2 , data)))

    # Calcular el error en la clase glándula:
    gland = len(list(filter(lambda a: a[0]==1 and a[1]==1 and a[2]==0, data)))
    
    # Calcular el error en la clase estroma:
    stroma = len(list(filter(lambda a: a[0]==0 and a[1]==0 and a[2]==1, data)))

    # Calcular el error en el estroma contaminado:
    contaminated = len(list(filter(lambda a: a[0]==2 and a[1]==0 and a[2]==1, data)))

    print("Total de errores cometidos:      {:4d}/{:4d} ({:.2f}%)"
        .format(N, y_orig.shape[0], ((N / y_orig.shape[0]) * 100)))
    print("Errores en la clase Glándula:    {:4d}/{:4d} ({:.2f}%, {:.2f}% del error total)"
        .format(gland, N_gland, ((gland / N_gland) * 100), ((gland / N) * 100)))
    print("Errores en la clase Estroma:     {:4d}/{:4d} ({:.2f}%, {:.2f}% del error total)"
        .format(stroma, N_stroma, ((stroma / N_stroma) * 100), ((stroma / N) * 100)))
    print("Errores en la clase Contaminado: {:4d}/{:4d} ({:.2f}%, {:.2f}% del error total)"
        .format(contaminated, N_contaminated, ((contaminated / N_contaminated) * 100), 
        ((contaminated / N) * 100)))


#########################################################################################################


def getFPandFN(predict, true):
    """Función auxiliar que será vectorizada para extraer los falsos positivos
    y falsos negativos y poder dibujarlos posteriormente.
    """
    if predict == true:
        return 0
    elif predict == 0 and true == 1:
        # Falso negativo:
        return 2
    else:
        # Falso positivo:
        return 1

vectFPFN = np.vectorize(getFPandFN)


#########################################################################################################


### CREAR LAS DIVISIONES PARA VALIDACIÓN CRUZADA ###

IMG_FOLDER = '../../Datos/Train/Images/'
QPH_FOLDER = '../../Datos/Train/Qpdat/'

# Unir cada imagen con su archivo de QuPath:
data = zipImagesFiles(IMG_FOLDER, QPH_FOLDER)

# Crear las divisiones para validación cruzada:
kf = KFold(n_splits=5)

# Declarar las variables necesarias:
classes = ('Stroma', 'Glands')
cvscores = {'train':[], 'val':[]}
tprs = []
aucs = []
cvtprs = []
cvfprs = []
cvthresholds = []
mean_fpr = np.linspace(0, 1, 100)
cms = np.empty(shape=(2,2,0), dtype=np.int64)
iters = 1

# Comenzar a entrenar con validación cruzada:
for train_index, test_index in kf.split(data):
    
    print("\nPARTITION", iters, '\n')

    train = list(itemgetter(*train_index)(data)) 
    val   = list(itemgetter( *test_index)(data))

    # Construir el conjunto de train:
    X_train = []
    y_train = []
    
    for image in train:
        img = mpimg.imread(image[0])
        p, l, c = build_from_QuPath(img, image[1], {'Stroma':0, 'Gland':1, 'Contaminated':2}, 32)
        p = add_scale(img, p, c, 32, 128)
        p, l = data_augmentation(p, l)
        X_train.append(p)
        y_train.append(l)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Construir el conjunto de validación:
    X_val = []
    y_val = []
    val_sizes = []
    c_val = []

    for image in val:
        img = mpimg.imread(image[0])
        p, l, c = build_from_QuPath(img, image[1], {'Stroma':0, 'Gland':1, 'Contaminated':2}, 32)
        p = add_scale(img, p, c, 32, 128)
        c_val.append(c)
        X_val.append(p)
        y_val.append(l)
        val_sizes.append(l.shape[0])

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Guardamos las etiquetas de validación originales:
    y_val_orig = y_val.copy()

    # Convertir el problema en binario:
    y_train[y_train==2] = 0
    y_val[y_val==2] = 0

    ### PREPROCESADO ###

    # Dado que los datos están desbalanceados, calcularemos el peso de cada clase:
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    print("\nClass Weights:", class_weights, '\n')

    # Normalizar los conjuntos de train y val:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train -= mean
    X_train /= std

    X_val -= mean
    X_val /= std

    # Aplicar one hot encoder a las etiquetas:
    y_train_ohe = to_categorical(y_train)
    y_val_ohe = to_categorical(y_val)

    ### ENTRENAMIENTO ###

    BATCH = 64
    EPOCHS = 100

    # Definir el modelo:
    model = GlandsDetector(input_shape=(32,32,6), output=2)

    # Definir el optimizador:
    adam = Adam()
    
    # Compilar el modelo:
    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=['binary_accuracy', km.binary_f1_score(label=1), 
            km.binary_precision(label=1), km.binary_recall(label=1)]
    )

    callback_list = [LearningRateScheduler(step_decay)]

    # Entrenar:
    history = model.fit(
        X_train, 
        y_train_ohe, 
        batch_size=BATCH, 
        epochs=EPOCHS, 
        validation_data=(X_val, y_val_ohe),
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=2
    )

    # Crear y guardar las curvas de aprendizaje:
    x_axis = np.arange(1,EPOCHS+1, dtype=np.int)
    f, axs = plt.subplots(2,2, figsize=(12, 10))
    axs = axs.flatten()

    # Dibujar la curva de error y de accuracy:
    axs[0].plot(x_axis, np.array(history.history['loss']), label='Train Loss')
    axs[0].plot(x_axis, np.array(history.history['val_loss']), label='Val Loss')
    axs[0].plot(x_axis, np.array(history.history['binary_accuracy']), label='Train Acc')
    axs[0].plot(x_axis, np.array(history.history['val_binary_accuracy']), label='Val Acc')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("Loss and accuracy over epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss / Accuracy")
    axs[0].legend()

    # Dibujar el F1 score, precision y recall:
    axs[1].plot(x_axis, np.array(history.history['f1_score']), alpha=0.3, label='Train F1')
    axs[1].plot(x_axis, np.array(history.history['val_f1_score']), label='Val F1')
    axs[1].plot(x_axis, np.array(history.history['precision']), alpha=0.3, label='Train Prec')
    axs[1].plot(x_axis, np.array(history.history['val_precision']), label='Val Prec')
    axs[1].plot(x_axis, np.array(history.history['recall']), alpha=0.3, label='Train Rec')
    axs[1].plot(x_axis, np.array(history.history['val_recall']), label='Val Rec')
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("F1 score, precision and recall over epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("F1 / Precision / Recall")
    axs[1].legend()

    # Matriz de confusión:
    y_pred = model.predict_classes(X_val)
    cm = confusion_matrix(y_val, y_pred)
    im = axs[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[2].figure.colorbar(im, ax=axs[2])
    axs[2].set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[2].text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    

    # Curva ROC:
    scores = model.predict(X_val)[:,1] 
    fpr, tpr, thresholds = roc_curve(y_val, scores)
    axs[3].plot([0, 1], [0, 1], 'k--', label='Chance')
    roc_auc = auc(fpr, tpr)
    axs[3].plot(fpr, tpr, label='Area = {:.4f}'.format(roc_auc))
    axs[3].set_title('ROC curve over validation data for Gland class')
    axs[3].set_xlabel('False positive rate')
    axs[3].set_ylabel('True positive rate')
    axs[3].legend()

    plt.suptitle("Model train and validation performance: crossval. partition " + str(iters), fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Train_curve_" + str(iters) + '.png')

    # Calcular las métricas para las estadísticas finales:
    cvthresholds.append(find_optimal_cuttoff(fpr, tpr, thresholds))
    cvtprs.append(tpr)
    cvfprs.append(fpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    aucs.append(roc_auc)
    tprs[-1][0] = 0.0

    cms = np.concatenate((cms, cm.reshape(2,2,1)), axis=2)

    scores = model.evaluate(X_train, y_train_ohe, verbose=0)
    cvscores['train'].append(scores)

    scores = model.evaluate(X_val, y_val_ohe, verbose=0)
    cvscores['val'].append(scores)

    # Dibujamos los falsos positivos y falsos negativos:
    if define_compute_FPFN:
        start = 0
        for i,size in enumerate(val_sizes):
            image = mpimg.imread(val[i][0])
            p = X_val[start:start+size]
            l = y_val[start:start+size]
            c = c_val[i]
            start += size
            # Predecir:
            predict = model.predict_classes(p)
            # Obtener los FP y FN:
            res = vectFPFN(predict, l)
            # Dibujar en azul los FP y en rojo los FN:
            draw_patches_of_class(image, c, res, {1:(0,0,255), 2:(255,0,0)}, 
                "valit_" + str(iters) + "_" + str(i) + ".png", 32)

    # Mostrar el porcentaje de error:
    error_summary(y_val_orig, y_val, model.predict_classes(X_val))
    
    iters += 1

##########################################################################################################

### CALCULAR ESTADÍSTICAS FINALES ###

# Calcular la curva roc media y la matriz de confusión final:
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

final_cm = np.sum(cms, axis=2)

f, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot([0, 1], [0, 1], 'k--', label='Chance')

# Plot de las curvas ROC de cada fold:
for i in range(5):
    ax1.plot(cvfprs[i], cvtprs[i], lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i+1, aucs[i]))

# Plot de la curva media:
ax1.plot(mean_fpr, mean_tpr, color='b',
    label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=2, alpha=.8)

# Std
ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC curve')
ax1.legend(loc="lower right")

im = ax2.imshow(final_cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.figure.colorbar(im, ax=ax2)
ax2.set(xticks=np.arange(final_cm.shape[1]),
    yticks=np.arange(final_cm.shape[0]),
    xticklabels=classes, yticklabels=classes,
    title='Confusion Matrix',
    ylabel='True label',
    xlabel='Predicted label')
thresh = final_cm.max() / 2.
for i in range(final_cm.shape[0]):
    for j in range(final_cm.shape[1]):
        ax2.text(j, i, format(final_cm[i, j], 'd'), ha="center", va="center",
            color="white" if final_cm[i, j] > thresh else "black")

plt.suptitle("Mean ROC curve and final confusion matrix", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ROC_cm.png')


# Mostrar media y std de las demás métricas:
means_train = np.mean(np.array(cvscores['train']), axis=0)
stds_train = np.std(np.array(cvscores['train']), axis=0)
means_val = np.mean(np.array(cvscores['val']), axis=0)
stds_val = np.std(np.array(cvscores['val']), axis=0)

print("")

# Umbral óptimo:
print("Optimal Cuttoff: %f +- %f" % (np.mean(cvthresholds), np.std(cvthresholds)))

print("Loss Train: %.4f +- %.4f" % (means_train[0], stds_train[0]))
print("Loss Valid: %.4f +- %.4f" % (means_val[0], stds_val[0]))

print("Accuracy Train: %.4f +- %.4f" % (means_train[1], stds_train[1]))
print("Accuracy Valid: %.4f +- %.4f" % (means_val[1], stds_val[1]))

print("F1 Score Train: %.4f +- %.4f" % (means_train[2], stds_train[2]))
print("F1 Score Valid: %.4f +- %.4f" % (means_val[2], stds_val[2]))

print("Precision Train: %.4f +- %.4f" % (means_train[3], stds_train[3]))
print("Precision Valid: %.4f +- %.4f" % (means_val[3], stds_val[3]))

print("Recall Train: %.4f +- %.4f" % (means_train[4], stds_train[4]))
print("Recall Valid: %.4f +- %.4f" % (means_val[4], stds_val[4]))
