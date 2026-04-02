import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

model = models.resnet18(weights='IMAGENET1K_V1')
model = model.eval()

test_img = Image.open(r'D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\cat.jpg')
test_img_data = np.asarray(test_img)
# plt.imshow(test_img_data)
# plt.show()

# model expects 224x224 3-color image
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

labels_path = r'D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[pred_label_idx.item()]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# #使用集成梯度进行特征归因
# # Initialize the attribution algorithm with the model
# integrated_gradients = IntegratedGradients(model)
#
# # Ask the algorithm to attribute our output target to
# attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)
#
# # Show the original image for comparison
# # 修复后：正确显示原始图片
# _ = viz.visualize_image_attr(
#     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),  # 传入原图数据
#     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
#     method="original_image",
#     title="Original Image"
# )

# default_cmap = LinearSegmentedColormap.from_list('custom blue',
#                                                  [(0, '#ffffff'),
#                                                   (0.25, '#0000ff'),
#                                                   (1, '#0000ff')], N=256)
#
# _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
#                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
#                              method='heat_map',
#                              cmap=default_cmap,
#                              show_colorbar=True,
#                              sign='positive',
#                              title='Integrated Gradients')



# #使用遮挡进行特征归因
# occlusion = Occlusion(model)
#
# attributions_occ = occlusion.attribute(input_img,
#                                        target=pred_label_idx,
#                                        strides=(3, 8, 8),
#                                        sliding_window_shapes=(3,15, 15),
#                                        baselines=0)
#
#
# _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       ["original_image", "heat_map", "heat_map", "masked_image"],
#                                       ["all", "positive", "negative", "positive"],
#                                       show_colorbar=True,
#                                       titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
#                                       fig_size=(18, 6)
#                                      )
#
#
# #使用 Layer GradCAM 进行层归因
# layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
# attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)
#
# _ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
#                              sign="all",
#                              title="Layer 3 Block 1 Conv 2")


#使用 Captum Insights 进行可视化
imgs = [r'D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\cat.jpg', r'D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\dog.jpg', r'D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\ikun.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[pred_label_idx.item()][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')