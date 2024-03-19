import sys
sys.path.append('C:\\Users\\amine\\Documents\\Amine_Files\\PhD\\connected_github_repos\\experiment-scenic\\big_vision\\big_vision')
print(sys.path)
import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform



class ModelInitialization:
    def __init__(self):
        self.config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
        self.module = models.TextZeroShotDetectionModule(
            body_configs=self.config.model.body,
            objectness_head_configs=self.config.model.objectness_head,
            normalize=self.config.model.normalize,
            box_bias=self.config.model.box_bias)
        self.variables = self.module.load_variables(self.config.init_from.checkpoint_path)

    def get_module(self):
        return self.module

    def get_variables(self):
        return self.variables

class ImageProcessor:
    @staticmethod
    def load_and_preprocess_image(filename, input_size):
        image_uint8 = skimage_io.imread(filename)
        if image_uint8.shape[-1] == 4:  # Convert RGBA to RGB
            image_uint8 = image_uint8[..., :3]
        image = image_uint8.astype(np.float32) / 255.0

        # Padding
        h, w, _ = image.shape
        size = max(h, w)
        image_padded = np.pad(
            image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

        # Resizing
        input_image = skimage_transform.resize(
            image_padded, (input_size, input_size), anti_aliasing=True)
        
        return input_image

class QueryProcessor:
    @staticmethod
    def tokenize_queries(module, text_queries, max_query_length):
        tokenized_queries = np.array([
            module.tokenize(q, max_query_length)
            for q in text_queries
        ])
        
        # Padding
        tokenized_queries = np.pad(
            tokenized_queries,
            pad_width=((0, 100 - len(text_queries)), (0, 0)),
            constant_values=0)
        return tokenized_queries

class Predictor:
    def __init__(self, module, variables):
        self.module = module
        self.variables = variables
        self.jitted = jax.jit(module.apply, static_argnames=('train',))

    def predict(self, image, tokenized_queries):
        predictions = self.jitted(
            self.variables,
            image[None, ...],
            tokenized_queries[None, ...],
            train=False)
        return jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)

class Visualizer:
    @staticmethod
    def visualize_predictions(input_image, predictions, text_queries, score_threshold=0.2):
        logits = predictions['pred_logits'][..., :len(text_queries)]
        scores = sigmoid(np.max(logits, axis=-1))
        labels = np.argmax(predictions['pred_logits'], axis=-1)
        boxes = predictions['pred_boxes']

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        for score, box, label in zip(scores, boxes, labels):
            if score < score_threshold:
                continue
            cx, cy, w, h = box
            ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                    [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
            ax.text(
                cx - w / 2,
                cy + h / 2 + 0.015,
                f'{text_queries[label]}: {score:.2f}',
                ha='left',
                va='top',
                color='red',
                bbox={'facecolor': 'white', 'edgecolor': 'red', 'boxstyle': 'square,pad=.3'})


if __name__ == '__main__':
    # Initialize the model with the config path (assuming a function or a path variable exists)
    model_init = ModelInitialization()

    # Get the initialized module and variables for the model
    module = model_init.get_module()
    variables = model_init.get_variables()

    # Assuming the input size and max query length are specified in the config
    input_size = model_init.config.dataset_configs.input_size
    max_query_length = model_init.config.dataset_configs.max_query_length

    # Process the image
    image_processor = ImageProcessor()
    input_image = image_processor.load_and_preprocess_image('./assets/sample_1.png', input_size)

    # Prepare the text queries
    query_processor = QueryProcessor()
    text_queries = ['face', 'man', 'woman', 'gun']
    tokenized_queries = query_processor.tokenize_queries(module, text_queries, max_query_length)

    # Make predictions
    predictor = Predictor(module, variables)
    predictions = predictor.predict(input_image, tokenized_queries)

    # Visualize the predictions
    Visualizer.visualize_predictions(input_image, predictions, text_queries)
