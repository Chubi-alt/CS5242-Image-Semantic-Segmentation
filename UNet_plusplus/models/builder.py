import torch
import segmentation_models_pytorch as smp
from .my_unetpp import MyUnetPlusPlus
import segmentation_models_pytorch as smp

import segmentation_models_pytorch as smp
from .my_unetpp import MyUnetPlusPlus # Ensure your hand-rolled file is named my_unetpp.py

def build_unetplusplus(backbone_name="resnet34", 
                       encoder_weights="imagenet", 
                       in_channels=3, 
                       num_classes=32, 
                       decoder_dropout=0.3,
                       model_type="smp"):
    """
    Builds a UNet++ model. Supports both SMP library and hand-rolled version.
    
    Args:
        backbone_name (str): Backbone for SMP model (ignored if model_type='scratch').
        encoder_weights (str): Pre-training weights for SMP.
        in_channels (int): Input image channels.
        num_classes (int): Number of segmentation classes.
        model_type (str): 'smp' for library model, 'scratch' for hand-rolled model.
    """
    
    if model_type == "scratch":
        print("[*] Building hand-rolled UNet++ (from scratch)...")
        # Ensure MyUnetPlusPlus handles in_channels and num_classes correctly
        model = MyUnetPlusPlus(in_channels=in_channels, num_classes=num_classes)
        
    else:
        print(f"[*] Building SMP UNet++ with {backbone_name} backbone...")
        model = smp.UnetPlusPlus(
            encoder_name=backbone_name,        
            encoder_weights=encoder_weights,     
            in_channels=in_channels,                  
            classes=num_classes,
            dropout=decoder_dropout                      
        )
    
    return model

if __name__ == '__main__':
    """
    A quick test block to verify that the model initializes correctly 
    and produces the expected output tensor shape.
    """
    # Note: Adjust num_classes to match the exact number of rows in your class_dict.csv
    # CamVid usually has 32 classes, but often 11 or 12 are used for training. 
    # Let's assume 32 for this quick test.
    test_classes = 32
    
    print(f"Building UNet++ with ResNet34 backbone for {test_classes} classes...")
    model = build_unetplusplus(backbone_name="resnet34", num_classes=test_classes)
    
    # Create a dummy input tensor: (Batch_Size, Channels, Height, Width)
    # Using 512x512 as your teammate specified in the evaluation command
    dummy_input = torch.randn(2, 3, 512, 512)
    
    print("Running a forward pass...")
    # Do not compute gradients for this simple test
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test successful!")