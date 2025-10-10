import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import shap
from lime import lime_image

# Load model
model = tf.keras.models.load_model('../results/cnn_pneumonia_model.h5')

# Load a test image
img_path = '../data/test/PNEUMONIA/person1_virus_6.jpeg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150,150))
x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# ---------------- SHAP Explanation ----------------
explainer_shap = shap.GradientExplainer(model, x)
shap_values = explainer_shap.shap_values(x)
os.makedirs('../results/shap_explanations', exist_ok=True)
shap.image_plot(shap_values, x, show=False)
plt.savefig('../results/shap_explanations/shap_result.png')
plt.close()

# ---------------- LIME Explanation ----------------
explainer_lime = lime_image.LimeImageExplainer()
explanation = explainer_lime.explain_instance(
    x[0].astype('double'), 
    model.predict, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000
)
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.axis('off')
os.makedirs('../results/lime_explanations', exist_ok=True)
plt.savefig('../results/lime_explanations/lime_result.png')
plt.close()

print("SHAP and LIME explanations generated successfully.")
