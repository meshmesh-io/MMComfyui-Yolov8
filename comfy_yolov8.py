import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class MMYolov8DetectionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    FUNCTION = "detect"
    CATEGORY = "yolov8"

    def detect(self, image, model_name):
        # Convert tensor to numpy array and then to PIL Image
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image
        
        print(f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')  # load a custom model
        results = model(image)

        # TODO load masks
        # masks = results[0].masks

        im_array = results[0].plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[...,::-1])  # RGB PIL image

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)  # Convert back to CxHxW
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        return (image_tensor_out, {"classify": [r.boxes.cls.tolist()[0] for r in results]})

# Helper function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class MMYolov8SegNode:
    def __init__(self) -> None:
        ...
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
                "class_id": ("INT", {"default": 0})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "JSON")  # Updated to include JSON for the color map return
    FUNCTION = "seg"
    CATEGORY = "yolov8"

    def seg(self, image, model_name, class_id):
        # Predefined set of 10 high-contrast hex colors
        colors_hex = [
            "#FF0000", # Red
            "#00FF00", # Lime
            "#0000FF", # Blue
            "#FFFF00", # Yellow
            "#FF00FF", # Magenta
            "#00FFFF", # Cyan
            "#FF8000", # Orange
            "#800080", # Purple
            "#008000", # Green
            "#000080", # Navy
        ]
        # Convert hex colors to RGB
        colors_rgb = [hex_to_rgb(hex_color) for hex_color in colors_hex]

        # Convert tensor to numpy array and then to PIL Image
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))

        print(f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        results = model(image)

        masks = results[0].masks.data
        boxes = results[0].boxes.data
        clss = boxes[:, 5]
        people_indices = torch.where(clss == class_id)
        people_masks = masks[people_indices]

        im_array = np.array(image).copy()
        color_map = {}  # Initialize the color map dictionary

        for i, mask in enumerate(people_masks):
            mask = mask > 0.5
            color = colors_rgb[i % len(colors_rgb)]
            color_map[i] = {'rgb': color, 'hex': colors_hex[i % len(colors_hex)]}  # Map person index to color

            for c in range(3):  # For RGB channels
                im_array[:, :, c] = np.where(mask, color[c], im_array[:, :, c])

        im_colored = Image.fromarray(im_array)
        image_tensor_out = torch.tensor(np.array(im_colored).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        combined_mask = torch.any(torch.stack(people_masks), dim=0).int() * 255

        return (image_tensor_out, combined_mask, color_map)

NODE_CLASS_MAPPINGS = {
    "MMYolov8Detection": MMYolov8DetectionNode,
    "MMYolov8Segmentation": MMYolov8SegNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMYolov8Detection": "detection",
    "MMYolov8Segmentation": "seg",
}

