import torch
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils.data import IDRiDDataset
from utils.transforms import build_idrid_transform
from loss import coral_predict
from models.res_unet_pp import ResUNetPPMultiTask


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_idrid_comparison(image_tensor, target_tensor, pred_tensor, threshold=0.5):
    """
    Args:
        image_tensor: [3, H, W] (Float 0-1)
        target_tensor: [4, H, W] (Binary 0/1)
        pred_tensor: [4, H, W] (Logits or Probs)
    """
    # 1. Prepare Image: Convert to Grayscale Numpy
    # [3, H, W] -> [H, W, 3] -> Grayscale [H, W]
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Contrast stretching for better visualization of vessels
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to 3-channel gray so we can add colored masks
    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) 

    # 2. Prepare Masks: [4, H, W] -> Numpy
    target_np = target_tensor.cpu().numpy()
    # Apply sigmoid if predictions are logits, then threshold
    pred_probs = torch.sigmoid(pred_tensor) if pred_tensor.max() > 1 else pred_tensor
    pred_np = (pred_probs > threshold).cpu().numpy().astype(np.float32)

    # 3. Define Clinical Colors (R, G, B)
    # Channel 0: MA (Red), 1: HE (Green), 2: EX (Blue), 3: SE (Yellow)
    colors = [
        [1, 0, 0], # Red
        [0, 1, 0], # Green
        [0, 0, 1], # Blue
        [1, 1, 0]  # Yellow
    ]

    def create_overlay(base_img, mask_stack, alpha=0.8):
        # 1. Ensure floating point 0-1 range
        if base_img.max() > 1.0:
            base_img = base_img.astype(np.float32) / 255.0
        
        # 2. Create a clean RGB copy of the grayscale base
        # This ensures the background exists everywhere, not just under masks
        combined_img = base_img.copy()

        for i, color in enumerate(colors):
            mask = mask_stack[i]
            if mask.max() == 0: continue
            
            # Reshape mask for broadcasting [H, W] -> [H, W, 1]
            mask_expanded = np.expand_dims(mask, axis=-1)
            
            # Standard Alpha Blending Formula: Out = (1-a)*Background + a*Foreground
            # By using the mask as a weight, non-mask areas stay 100% background
            # and mask areas become a blend of anatomy + lesion color
            color_layer = np.ones_like(combined_img) * np.array(color)
            
            combined_img = (1 - (mask_expanded * alpha)) * combined_img + (mask_expanded * alpha) * color_layer
                    
        return np.clip(combined_img, 0, 1)

    # Generate Overlays
    gt_overlay = create_overlay(gray_rgb, target_np)
    pr_overlay = create_overlay(gray_rgb, pred_np)

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=100)
    
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Original (Grayscale)", fontsize=20)
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth Masks", fontsize=20)
    
    axes[2].imshow(pr_overlay)
    axes[2].set_title(f"Predicted Masks (T={threshold})", fontsize=20)

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    save_path = "inference_result.png"
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.close(fig)


if __name__ == '__main__':

    model = ResUNetPPMultiTask('r50', 4, 5, 3, 0.3).to(device)
    ckpt = torch.load('runs/idrid_mt_r50_1024_union/last.pt', map_location=device, weights_only=True)

    model.eval()
    model.load_state_dict(ckpt['model'])


    test = IDRiDDataset(
        'data/archive',
        split='test',
        task_mode='multitask',
        require_both=True,
        transform=build_idrid_transform(
            train=False,
            image_size=ckpt['args']['image_size'], # 896
            use_clahe=False,
            )
        )

    img = test[0]['image']
    seg_mask = test[0]['seg_mask']
    with torch.inference_mode():
        # with torch.amp.autocast_mode.autocast(device, torch.float16, True):
        out = model(img.unsqueeze(0).to(device))

    print(
        f"GT DR-Grade: {test[0]['dr_grade'].item()}\n"
        f"Predicted DR-Grade: {coral_predict(out['dr_logits']).item()}\n"
        f"GT DME-Grade: {test[0]['dme_grade'].item()}\n"
        f"Predicted DME-Grade: {coral_predict(out['dme_logits']).item()}\n"
        )
    plot_idrid_comparison(img, seg_mask, out['seg_logits'].squeeze(0))
